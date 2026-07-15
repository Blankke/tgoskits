use alloc::{collections::VecDeque, sync::Arc};
use core::mem::MaybeUninit;
#[cfg(feature = "smp")]
use core::ptr::NonNull;
#[cfg(feature = "smp")]
use core::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

use ax_hal::percpu::this_cpu_id;
use ax_kernel_guard::BaseGuard;
use ax_kspin::{SpinNoIrqGuard, SpinRaw};
use ax_lazyinit::LazyInit;
use ax_memory_addr::VirtAddr;
use ax_sched::BaseScheduler;

use crate::{
    AxCpuMask, AxTaskRef, Scheduler, TaskInner, WaitQueue,
    task::{CurrentTask, TASK_STACK_ALIGN, TaskStack, TaskState},
    wait_queue::WaitQueueGuard,
};

macro_rules! percpu_static {
    ($(
        $(#[$comment:meta])*
        $name:ident: $ty:ty = $init:expr
    ),* $(,)?) => {
        $(
            $(#[$comment])*
            #[ax_percpu::def_percpu]
            static $name: $ty = $init;
        )*
    };
}

percpu_static! {
    RUN_QUEUE: LazyInit<AxRunQueue> = LazyInit::new(),
    EXITED_TASKS: VecDeque<AxTaskRef> = VecDeque::new(),
    WAIT_FOR_EXIT: WaitQueue = WaitQueue::new(),
    IDLE_TASK: LazyInit<AxTaskRef> = LazyInit::new(),
    /// Stores a raw pointer to the previous task running on this CPU.
    /// The pointer is valid only within the window between `switch_to` storing it
    /// and `clear_prev_task_on_cpu` consuming it — both in the same non-preemptible
    /// call chain, so the task cannot be freed while the pointer is held.
    #[cfg(feature = "smp")]
    PREV_TASK: Option<NonNull<crate::AxTask>> = None,
}

/// An array of references to run queues, one for each CPU, indexed by cpu_id.
///
/// This static variable holds references to the run queues for each CPU in the system.
///
/// # Safety
///
/// Access to this variable is marked as `unsafe` because it contains `MaybeUninit` references,
/// which require careful handling to avoid undefined behavior. The array should be fully
/// initialized before being accessed to ensure safe usage.
static mut RUN_QUEUES: [MaybeUninit<&'static mut AxRunQueue>; crate::build_info::CPU_CAPACITY] =
    [ARRAY_REPEAT_VALUE; crate::build_info::CPU_CAPACITY];
#[allow(clippy::declare_interior_mutable_const)] // It's ok because it's used only for initialization `RUN_QUEUES`.
const ARRAY_REPEAT_VALUE: MaybeUninit<&'static mut AxRunQueue> = MaybeUninit::uninit();

/// Publishes which entries in `RUN_QUEUES` are safe to dereference.
#[cfg(feature = "smp")]
static RUN_QUEUE_INITIALIZED: [AtomicBool; crate::build_info::CPU_CAPACITY] =
    [const { AtomicBool::new(false) }; crate::build_info::CPU_CAPACITY];

/// A non-idle CPU whose current task contributes one unit of runnable load.
#[cfg(feature = "smp")]
const RUN_QUEUE_ACTIVITY_BUSY: u8 = 0;
/// An idle CPU that can be atomically reserved by one placement operation.
#[cfg(feature = "smp")]
const RUN_QUEUE_ACTIVITY_IDLE: u8 = 1;
/// An idle CPU reserved for a task that has not committed its enqueue yet.
#[cfg(feature = "smp")]
const RUN_QUEUE_ACTIVITY_RESERVED: u8 = 2;

/// Tracks whether each CPU is busy, idle, or reserved by one pending enqueue.
///
/// This is a placement hint, not scheduler state. The reserved state prevents
/// concurrent fork/wake bursts from selecting the same idle CPU, while still
/// allowing a failed wakeup CAS to return its reservation safely.
#[cfg(feature = "smp")]
static RUN_QUEUE_ACTIVITY: [AtomicU8; crate::build_info::CPU_CAPACITY] =
    [const { AtomicU8::new(RUN_QUEUE_ACTIVITY_BUSY) }; crate::build_info::CPU_CAPACITY];

#[cfg(feature = "smp")]
#[inline]
fn release_idle_run_queue_reservation_with(activity: &AtomicU8) -> bool {
    activity
        .compare_exchange(
            RUN_QUEUE_ACTIVITY_RESERVED,
            RUN_QUEUE_ACTIVITY_IDLE,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok()
}

#[cfg(feature = "smp")]
#[inline]
fn release_idle_run_queue_reservation(cpu_id: usize) {
    let _ = release_idle_run_queue_reservation_with(&RUN_QUEUE_ACTIVITY[cpu_id]);
}

/// Independent cursors keep idle scans separate from load-tie and migration
/// round-robin ordering.
#[cfg(feature = "smp")]
static NEXT_IDLE_RUN_QUEUE: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "smp")]
static NEXT_RUN_QUEUE: AtomicUsize = AtomicUsize::new(0);

#[cfg(not(feature = "host-test"))]
fn main_task_stack() -> TaskStack {
    let (stack_ptr, stack_size) = ax_hal::mem::boot_stack_bounds(this_cpu_id());
    TaskStack::borrowed(stack_ptr, stack_size, TASK_STACK_ALIGN)
}

#[cfg(feature = "host-test")]
fn main_task_stack() -> TaskStack {
    TaskStack::alloc(crate::default_task_stack_size())
}

/// Returns a reference to the current run queue in [`CurrentRunQueueRef`].
///
/// ## Safety
///
/// This function returns a static reference to the current run queue, which
/// is inherently unsafe. It assumes that the `RUN_QUEUE` has been properly
/// initialized and is not accessed concurrently in a way that could cause
/// data races or undefined behavior.
///
/// ## Returns
///
/// * [`CurrentRunQueueRef`] - a static reference to the current [`AxRunQueue`].
#[inline(always)]
pub(crate) fn current_run_queue<G: BaseGuard>() -> CurrentRunQueueRef<'static, G> {
    let irq_state = G::acquire();
    CurrentRunQueueRef {
        inner: unsafe { RUN_QUEUE.current_ref_mut_raw() },
        current_task: crate::current(),
        state: irq_state,
        _phantom: core::marker::PhantomData,
    }
}

#[cfg(feature = "smp")]
#[inline]
fn online_cpu_count() -> usize {
    ax_hal::cpu_num().min(crate::build_info::CPU_CAPACITY)
}

#[cfg(feature = "smp")]
#[inline]
fn run_queue_accepts_tasks(cpu_id: usize) -> bool {
    if cpu_id >= online_cpu_count() || !RUN_QUEUE_INITIALIZED[cpu_id].load(Ordering::Acquire) {
        return false;
    }

    #[cfg(all(feature = "ipi", not(feature = "host-test")))]
    if cpu_id != this_cpu_id() && !ax_ipi::is_cpu_ready(cpu_id) {
        return false;
    }

    true
}

/// Finds the first matching CPU while scanning a bounded circular CPU range.
///
/// `start_cpu` is only a scan offset. It must never be treated as a placement
/// preference, because a creator CPU often launches an entire worker burst.
#[cfg(feature = "smp")]
#[inline]
fn find_run_queue_index(
    cpumask: AxCpuMask,
    start_cpu: usize,
    cpu_count: usize,
    mut matches: impl FnMut(usize) -> bool,
) -> Option<usize> {
    assert!(!cpumask.is_empty(), "No available CPU for task execution");
    if cpu_count == 0 {
        return None;
    }

    for offset in 0..cpu_count {
        let cpu_id = start_cpu.wrapping_add(offset) % cpu_count;
        if cpumask.get(cpu_id) && matches(cpu_id) {
            return Some(cpu_id);
        }
    }
    None
}

#[cfg(feature = "smp")]
#[inline]
fn select_run_queue_index(cpumask: AxCpuMask) -> usize {
    let cpu_count = online_cpu_count();
    let start_cpu = NEXT_RUN_QUEUE.fetch_add(1, Ordering::Relaxed);
    find_run_queue_index(cpumask, start_cpu, cpu_count, run_queue_accepts_tasks)
        .expect("No initialized run queue matches the task CPU affinity")
}

/// Selects the least loaded eligible CPU with deterministic locality ties.
///
/// `preferred_cpus` is ordered from strongest to weakest preference. It only
/// breaks equal-load ties; it never overrides a less loaded run queue.
#[cfg(feature = "smp")]
#[inline]
fn select_least_loaded_run_queue_index_with(
    cpumask: AxCpuMask,
    preferred_cpus: &[usize],
    start_cpu: usize,
    cpu_count: usize,
    mut is_ready: impl FnMut(usize) -> bool,
    mut runnable_load: impl FnMut(usize) -> usize,
) -> Option<usize> {
    assert!(!cpumask.is_empty(), "No available CPU for task execution");
    if cpu_count == 0 {
        return None;
    }

    let mut best_cpu = None;
    let mut best_load = usize::MAX;
    let mut best_preference = usize::MAX;
    for offset in 0..cpu_count {
        let cpu_id = start_cpu.wrapping_add(offset) % cpu_count;
        if !cpumask.get(cpu_id) || !is_ready(cpu_id) {
            continue;
        }

        let load = runnable_load(cpu_id);
        let preference = preferred_cpus
            .iter()
            .position(|preferred| *preferred == cpu_id)
            .unwrap_or(preferred_cpus.len());
        if load < best_load || (load == best_load && preference < best_preference) {
            best_cpu = Some(cpu_id);
            best_load = load;
            best_preference = preference;
        }
    }
    best_cpu
}

/// Selects a CPU for a newly created task.
///
/// This helper is intentionally independent from the per-CPU storage so host
/// tests can verify policy without pretending that one host-test CPU is
/// several real CPUs.
#[cfg(feature = "smp")]
#[inline]
fn select_new_task_run_queue_index_with(
    cpumask: AxCpuMask,
    current_cpu: usize,
    idle_start_cpu: usize,
    fallback_start_cpu: usize,
    cpu_count: usize,
    mut is_ready: impl FnMut(usize) -> bool,
    mut claim_idle: impl FnMut(usize) -> bool,
    runnable_load: impl FnMut(usize) -> usize,
) -> Option<RunQueueSelection> {
    if let Some(cpu_id) = find_run_queue_index(cpumask, idle_start_cpu, cpu_count, |cpu_id| {
        is_ready(cpu_id) && claim_idle(cpu_id)
    }) {
        return Some(RunQueueSelection {
            cpu_id,
            claimed_idle: true,
        });
    }

    // A fork burst commonly leaves the creator blocked after launching its
    // workers. Prefer it on an equal-load tie so the final worker does not
    // collide with an already claimed remote CPU and leave the creator idle.
    select_least_loaded_run_queue_index_with(
        cpumask,
        &[current_cpu],
        fallback_start_cpu,
        cpu_count,
        is_ready,
        runnable_load,
    )
    .map(|cpu_id| RunQueueSelection {
        cpu_id,
        claimed_idle: false,
    })
}

#[cfg(feature = "smp")]
#[inline]
fn run_queue_runnable_load(cpu_id: usize) -> usize {
    let ready_tasks = get_run_queue(cpu_id).ready_tasks.load(Ordering::Relaxed);
    let running_tasks =
        usize::from(RUN_QUEUE_ACTIVITY[cpu_id].load(Ordering::Acquire) != RUN_QUEUE_ACTIVITY_IDLE);
    ready_tasks + running_tasks
}

#[cfg(feature = "smp")]
#[inline]
fn try_claim_idle_run_queue(cpu_id: usize) -> bool {
    if get_run_queue(cpu_id).ready_tasks.load(Ordering::Relaxed) != 0 {
        return false;
    }
    RUN_QUEUE_ACTIVITY[cpu_id]
        .compare_exchange(
            RUN_QUEUE_ACTIVITY_IDLE,
            RUN_QUEUE_ACTIVITY_RESERVED,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok()
}

#[cfg(feature = "smp")]
#[inline]
fn select_new_task_run_queue_index(cpumask: AxCpuMask) -> RunQueueSelection {
    let cpu_count = online_cpu_count();
    let idle_start_cpu = NEXT_IDLE_RUN_QUEUE.fetch_add(1, Ordering::Relaxed);
    let fallback_start_cpu = NEXT_RUN_QUEUE.fetch_add(1, Ordering::Relaxed);

    select_new_task_run_queue_index_with(
        cpumask,
        this_cpu_id(),
        idle_start_cpu,
        fallback_start_cpu,
        cpu_count,
        run_queue_accepts_tasks,
        try_claim_idle_run_queue,
        run_queue_runnable_load,
    )
    .expect("No initialized run queue matches the task CPU affinity")
}

#[cfg(feature = "smp")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RunQueueSelection {
    cpu_id: usize,
    claimed_idle: bool,
}

#[cfg(feature = "smp")]
#[inline]
fn select_wake_run_queue_index_with(
    cpumask: AxCpuMask,
    current_cpu: usize,
    last_cpu: usize,
    idle_start_cpu: usize,
    fallback_start_cpu: usize,
    cpu_count: usize,
    mut is_ready: impl FnMut(usize) -> bool,
    mut claim_idle: impl FnMut(usize) -> bool,
    runnable_load: impl FnMut(usize) -> usize,
) -> Option<RunQueueSelection> {
    if last_cpu < cpu_count && cpumask.get(last_cpu) && is_ready(last_cpu) && claim_idle(last_cpu) {
        return Some(RunQueueSelection {
            cpu_id: last_cpu,
            claimed_idle: true,
        });
    }

    if let Some(cpu_id) = find_run_queue_index(cpumask, idle_start_cpu, cpu_count, |cpu_id| {
        is_ready(cpu_id) && claim_idle(cpu_id)
    }) {
        return Some(RunQueueSelection {
            cpu_id,
            claimed_idle: true,
        });
    }

    select_least_loaded_run_queue_index_with(
        cpumask,
        &[last_cpu, current_cpu],
        fallback_start_cpu,
        cpu_count,
        is_ready,
        runnable_load,
    )
    .map(|cpu_id| RunQueueSelection {
        cpu_id,
        claimed_idle: false,
    })
}

#[cfg(all(test, feature = "smp", feature = "host-test"))]
fn placement_test_cpu_count() -> usize {
    crate::build_info::CPU_CAPACITY.min(4)
}

#[cfg(all(test, feature = "smp", feature = "host-test"))]
#[test]
fn wake_placement_releases_a_failed_idle_reservation() {
    let activity = AtomicU8::new(RUN_QUEUE_ACTIVITY_RESERVED);

    assert!(release_idle_run_queue_reservation_with(&activity));
    assert_eq!(activity.load(Ordering::Acquire), RUN_QUEUE_ACTIVITY_IDLE);

    activity.store(RUN_QUEUE_ACTIVITY_BUSY, Ordering::Release);
    assert!(!release_idle_run_queue_reservation_with(&activity));
    assert_eq!(activity.load(Ordering::Acquire), RUN_QUEUE_ACTIVITY_BUSY);
}

#[cfg(all(test, feature = "smp", feature = "host-test"))]
#[test]
fn new_task_placement_uses_an_available_idle_cpu() {
    let cpu_count = placement_test_cpu_count();
    if cpu_count < 2 {
        return;
    }

    let mut cpumask = AxCpuMask::new();
    for cpu_id in 0..cpu_count {
        cpumask.set(cpu_id, true);
    }
    let idle_cpu = cpu_count - 1;

    let selected = select_new_task_run_queue_index_with(
        cpumask,
        0,
        0,
        0,
        cpu_count,
        |_| true,
        |cpu_id| cpu_id == idle_cpu,
        |_| 0,
    )
    .unwrap();

    assert_eq!(
        selected.cpu_id, idle_cpu,
        "a new task must use an available idle CPU instead of pinning to its creator",
    );
}

#[cfg(all(test, feature = "smp", feature = "host-test"))]
#[test]
fn new_task_placement_balances_load_when_no_cpu_is_idle() {
    let cpu_count = placement_test_cpu_count();
    if cpu_count < 2 {
        return;
    }

    let mut cpumask = AxCpuMask::new();
    for cpu_id in 0..cpu_count {
        cpumask.set(cpu_id, true);
    }

    let mut selected_cpus = AxCpuMask::new();
    let mut runnable_load = [0usize; crate::build_info::CPU_CAPACITY];
    for fallback_start_cpu in 0..cpu_count {
        let cpu_id = select_new_task_run_queue_index_with(
            cpumask,
            0,
            0,
            fallback_start_cpu,
            cpu_count,
            |_| true,
            |_| false,
            |cpu_id| runnable_load[cpu_id],
        )
        .unwrap()
        .cpu_id;
        selected_cpus.set(cpu_id, true);
        runnable_load[cpu_id] += 1;
    }

    assert_eq!(
        selected_cpus, cpumask,
        "a worker burst must spread across all allowed CPUs when none is idle",
    );
}

#[cfg(all(test, feature = "smp", feature = "host-test"))]
#[test]
fn new_task_placement_uses_creator_after_remote_idle_slots_are_claimed() {
    let cpu_count = placement_test_cpu_count();
    if cpu_count < 2 {
        return;
    }

    let mut cpumask = AxCpuMask::new();
    for cpu_id in 0..cpu_count {
        cpumask.set(cpu_id, true);
    }

    let selected = select_new_task_run_queue_index_with(
        cpumask,
        0,
        1,
        1,
        cpu_count,
        |_| true,
        |_| false,
        |_| 1,
    );

    assert_eq!(
        selected.map(|selection| selection.cpu_id),
        Some(0),
        "after remote idle slots are claimed, the next worker must use its creator CPU",
    );
}

#[cfg(all(test, feature = "smp", feature = "host-test"))]
#[test]
fn new_task_placement_skips_unready_run_queues() {
    let cpu_count = placement_test_cpu_count();
    if cpu_count < 2 {
        return;
    }

    let mut cpumask = AxCpuMask::new();
    for cpu_id in 0..cpu_count {
        cpumask.set(cpu_id, true);
    }

    let selected = select_new_task_run_queue_index_with(
        cpumask,
        0,
        0,
        1,
        cpu_count,
        |cpu_id| cpu_id == 0,
        |_| false,
        |_| 0,
    );

    assert_eq!(
        selected.map(|selection| selection.cpu_id),
        Some(0),
        "early boot must not choose a secondary CPU before its run queue is ready",
    );
}

#[cfg(all(test, feature = "smp", feature = "host-test"))]
#[test]
fn wake_placement_preserves_the_task_last_cpu() {
    let cpu_count = placement_test_cpu_count();
    if cpu_count < 2 {
        return;
    }

    let mut cpumask = AxCpuMask::new();
    for cpu_id in 0..cpu_count {
        cpumask.set(cpu_id, true);
    }
    let last_cpu = cpu_count - 1;

    let selected = select_wake_run_queue_index_with(
        cpumask,
        0,
        last_cpu,
        0,
        0,
        cpu_count,
        |_| true,
        |cpu_id| cpu_id == last_cpu,
        |_| 1,
    );

    assert_eq!(
        selected.map(|selection| selection.cpu_id),
        Some(last_cpu),
        "a centralized waker must not pull a runnable task away from its last CPU",
    );
}

#[cfg(all(test, feature = "smp", feature = "host-test"))]
#[test]
fn wake_placement_does_not_reuse_an_already_claimed_last_cpu() {
    let cpu_count = placement_test_cpu_count();
    if cpu_count < 2 {
        return;
    }

    let mut cpumask = AxCpuMask::new();
    for cpu_id in 0..cpu_count {
        cpumask.set(cpu_id, true);
    }
    let last_cpu = cpu_count - 1;
    let alternative_cpu = (last_cpu + 1) % cpu_count;

    let selected = select_wake_run_queue_index_with(
        cpumask,
        0,
        last_cpu,
        alternative_cpu,
        alternative_cpu,
        cpu_count,
        |_| true,
        |cpu_id| cpu_id == alternative_cpu,
        |_| 1,
    );

    assert_eq!(
        selected.map(|selection| selection.cpu_id),
        Some(alternative_cpu),
        "a second wake must use another idle CPU after the task's last CPU was claimed",
    );
}

/// Retrieves a `'static` reference to the run queue corresponding to the given index.
///
/// This function asserts that the provided index is within the range of available CPUs
/// and returns a reference to the corresponding run queue.
///
/// ## Arguments
///
/// * `index` - The index of the run queue to retrieve.
///
/// ## Returns
///
/// A reference to the `AxRunQueue` corresponding to the provided index.
///
/// ## Panics
///
/// This function will panic if the index is out of bounds.
#[cfg(feature = "smp")]
#[inline]
fn get_run_queue(index: usize) -> &'static mut AxRunQueue {
    unsafe { RUN_QUEUES[index].assume_init_mut() }
}

#[cfg(all(feature = "smp", feature = "ipi"))]
#[cfg_attr(all(test, feature = "host-test"), allow(dead_code))]
fn request_current_reschedule() {
    clear_remote_reschedule_pending_for_current_cpu();
    #[cfg(all(feature = "preempt", feature = "host-test"))]
    if let Some(curr) = crate::current_may_uninit() {
        curr.set_force_resched_pending(true);
    }
    #[cfg(all(feature = "preempt", not(feature = "host-test")))]
    if crate::current_may_uninit().is_some() {
        CurrentRunQueueRef::<ax_kernel_guard::NoOp>::force_resched_from_irq();
    }
}

#[cfg(all(test, feature = "smp", feature = "ipi", feature = "host-test"))]
static REMOTE_RESCHEDULE_REQUESTS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

#[cfg(all(
    feature = "smp",
    feature = "ipi",
    not(all(test, feature = "host-test"))
))]
static REMOTE_RESCHEDULE_PENDING: [AtomicBool; crate::build_info::CPU_CAPACITY] =
    [const { AtomicBool::new(false) }; crate::build_info::CPU_CAPACITY];

#[cfg(all(test, feature = "smp", feature = "ipi", feature = "host-test"))]
static REMOTE_RESCHEDULE_PENDING: AtomicBool = AtomicBool::new(false);

#[cfg(all(feature = "smp", feature = "ipi"))]
pub(crate) fn clear_remote_reschedule_pending_for_current_cpu() {
    #[cfg(not(all(test, feature = "host-test")))]
    REMOTE_RESCHEDULE_PENDING[this_cpu_id()].store(false, Ordering::Release);
    #[cfg(all(test, feature = "host-test"))]
    REMOTE_RESCHEDULE_PENDING.store(false, Ordering::Release);
}

#[cfg(all(feature = "smp", feature = "ipi"))]
fn request_remote_reschedule_if_not_pending<F>(pending: &AtomicBool, request: F)
where
    F: FnOnce(),
{
    if !pending.swap(true, Ordering::AcqRel) {
        request();
    }
}

#[cfg(all(feature = "smp", feature = "ipi"))]
fn force_remote_reschedule_request<F>(pending: &AtomicBool, request: F)
where
    F: FnOnce(),
{
    pending.store(true, Ordering::Release);
    request();
}

#[cfg(all(
    feature = "smp",
    feature = "ipi",
    not(all(test, feature = "host-test"))
))]
fn request_remote_reschedule(cpu_id: usize) {
    request_remote_reschedule_if_not_pending(&REMOTE_RESCHEDULE_PENDING[cpu_id], || {
        ax_ipi::run_on_cpu(cpu_id, request_current_reschedule);
    });
}

#[cfg(all(
    feature = "smp",
    feature = "ipi",
    not(all(test, feature = "host-test"))
))]
fn force_remote_reschedule(cpu_id: usize) {
    force_remote_reschedule_request(&REMOTE_RESCHEDULE_PENDING[cpu_id], || {
        ax_ipi::run_on_cpu(cpu_id, request_current_reschedule);
    });
}

#[cfg(all(test, feature = "smp", feature = "ipi", feature = "host-test"))]
fn request_remote_reschedule(cpu_id: usize) {
    let _ = cpu_id;
    // Host tests run with one dummy CPU and a no-op send_ipi(), so record the
    // scheduler-visible request that a real ax-ipi callback would carry.
    request_remote_reschedule_if_not_pending(&REMOTE_RESCHEDULE_PENDING, || {
        REMOTE_RESCHEDULE_REQUESTS.fetch_add(1, Ordering::Release);
    });
}

#[cfg(all(test, feature = "smp", feature = "ipi", feature = "host-test"))]
fn force_remote_reschedule(cpu_id: usize) {
    let _ = cpu_id;
    force_remote_reschedule_request(&REMOTE_RESCHEDULE_PENDING, || {
        REMOTE_RESCHEDULE_REQUESTS.fetch_add(1, Ordering::Release);
    });
}

#[cfg(all(feature = "smp", feature = "ipi"))]
fn kick_remote_cpu(cpu_id: usize) {
    if cpu_id != this_cpu_id() {
        // axruntime's IPI handler only drains ax-ipi callbacks. A bare hardware
        // IPI can wake an idle CPU, but it does not ask a running remote CPU to
        // reschedule after a task is queued there.
        request_remote_reschedule(cpu_id);
    }
}

#[cfg(all(feature = "smp", feature = "ipi"))]
fn force_kick_remote_cpu(cpu_id: usize) {
    if cpu_id != this_cpu_id() {
        force_remote_reschedule(cpu_id);
    }
}

#[cfg(all(test, feature = "smp", feature = "ipi", feature = "host-test"))]
mod tests {
    use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    // Host-test mode collapses per-CPU state into process-global statics, so
    // keep the shared pending/count assertions in one test.
    #[test]
    fn remote_reschedule_request_is_coalesced_and_forced() {
        const REMOTE_CPU: usize = 1;

        super::REMOTE_RESCHEDULE_REQUESTS.store(0, Ordering::Release);
        super::REMOTE_RESCHEDULE_PENDING.store(false, Ordering::Release);

        super::kick_remote_cpu(REMOTE_CPU);

        assert_eq!(
            super::REMOTE_RESCHEDULE_REQUESTS.load(Ordering::Acquire),
            1,
            "remote CPU kicks must enqueue a scheduler-visible reschedule request",
        );
        super::kick_remote_cpu(REMOTE_CPU);

        assert_eq!(
            super::REMOTE_RESCHEDULE_REQUESTS.load(Ordering::Acquire),
            1,
            "remote CPU kicks should coalesce identical pending reschedule requests",
        );

        super::clear_remote_reschedule_pending_for_current_cpu();
        super::kick_remote_cpu(REMOTE_CPU);

        assert_eq!(
            super::REMOTE_RESCHEDULE_REQUESTS.load(Ordering::Acquire),
            2,
            "remote CPU kicks must be accepted again after the pending bit is cleared",
        );

        #[cfg(feature = "preempt")]
        crate::tests::run_in_test_scheduler(|| {
            let curr = crate::current();

            curr.set_preempt_pending(false);
            curr.set_force_resched_pending(false);
            super::REMOTE_RESCHEDULE_PENDING.store(true, Ordering::Release);

            super::request_current_reschedule();

            assert!(
                curr.force_resched_pending_for_test(),
                "remote IPI reschedule must request forced rotation",
            );
            assert!(
                !curr.preempt_pending_for_test(),
                "remote IPI reschedule must not rely on ordinary RR preemption",
            );
            assert!(
                !super::REMOTE_RESCHEDULE_PENDING.load(Ordering::Acquire),
                "remote IPI callback must clear the coalescing bit when it is delivered",
            );

            curr.set_force_resched_pending(false);
            curr.set_preempt_pending(false);
        });

        #[cfg(feature = "preempt")]
        {
            super::kick_remote_cpu(REMOTE_CPU);
            assert_eq!(
                super::REMOTE_RESCHEDULE_REQUESTS.load(Ordering::Acquire),
                3,
                "a delivered remote IPI must allow a later kick to enqueue a new callback",
            );
        }

        super::REMOTE_RESCHEDULE_PENDING.store(false, Ordering::Release);
        super::REMOTE_RESCHEDULE_REQUESTS.store(0, Ordering::Release);
    }

    #[test]
    fn forced_remote_reschedule_bypasses_stale_pending() {
        let pending = AtomicBool::new(true);
        let requests = AtomicUsize::new(0);

        super::force_remote_reschedule_request(&pending, || {
            requests.fetch_add(1, Ordering::Release);
        });

        assert_eq!(
            requests.load(Ordering::Acquire),
            1,
            "forced remote kicks must bypass stale pending coalescing",
        );

        super::request_remote_reschedule_if_not_pending(&pending, || {
            requests.fetch_add(1, Ordering::Release);
        });

        assert_eq!(
            requests.load(Ordering::Acquire),
            1,
            "ordinary remote kicks should still coalesce stale pending requests",
        );

        super::force_remote_reschedule_request(&pending, || {
            requests.fetch_add(1, Ordering::Release);
        });

        assert_eq!(
            requests.load(Ordering::Acquire),
            2,
            "forced remote kicks must not coalesce required migration reschedules",
        );
    }
}

#[cfg(all(test, feature = "sched-rr", feature = "host-test"))]
mod rr_tests {
    use alloc::{string::String, sync::Arc};
    use core::marker::PhantomData;

    use ax_sched::BaseScheduler;

    use super::{AxRunQueue, AxRunQueueRef, Scheduler, SpinRaw, TaskInner};
    use crate::task::TaskState;

    fn new_test_task(name: &str, state: TaskState) -> crate::AxTaskRef {
        let task =
            TaskInner::new(|| {}, String::from(name), crate::default_task_stack_size()).into_arc();
        task.set_state(state);
        task
    }

    #[test]
    fn unblock_resched_does_not_front_insert_rr_task() {
        let mut run_queue = AxRunQueue {
            cpu_id: 1,
            scheduler: SpinRaw::new(Scheduler::new()),
        };
        let queued = new_test_task("queued", TaskState::Ready);
        let blocked = new_test_task("blocked", TaskState::Blocked);

        run_queue.scheduler.lock().add_task(queued.clone());
        {
            let mut run_queue_ref = AxRunQueueRef::<ax_kernel_guard::NoOp> {
                inner: &mut run_queue,
                state: (),
                _phantom: PhantomData,
            };
            run_queue_ref.unblock_task(blocked, true);
        }

        let next = run_queue.scheduler.lock().pick_next_task().unwrap();
        assert!(
            Arc::ptr_eq(&next, &queued),
            "waking a blocked task with resched=true must not move it ahead of already queued RR \
             tasks",
        );
    }
}

/// Selects a run queue for a newly created task.
///
/// The SMP path first reserves an eligible idle CPU, then selects the smallest
/// ready-plus-running load among initialized CPUs. Fork placement deliberately
/// differs from wake placement: a creator can launch a burst of workers, while
/// a blocked task benefits from returning to its previous CPU.
#[inline]
pub(crate) fn select_new_task_run_queue<G: BaseGuard>(
    task: &AxTaskRef,
) -> AxRunQueueRef<'static, G> {
    let irq_state = G::acquire();
    #[cfg(not(feature = "smp"))]
    {
        let _ = task;
        // When SMP is disabled, all tasks are scheduled on the same global run queue.
        AxRunQueueRef {
            inner: unsafe { RUN_QUEUE.current_ref_mut_raw() },
            state: irq_state,
            _phantom: core::marker::PhantomData,
        }
    }
    #[cfg(feature = "smp")]
    {
        let selection = select_new_task_run_queue_index(task.cpumask());
        AxRunQueueRef {
            inner: get_run_queue(selection.cpu_id),
            state: irq_state,
            idle_reservation: selection.claimed_idle,
            _phantom: core::marker::PhantomData,
        }
    }
}

/// Selects a run queue for waking a blocked task.
///
/// Wakeups atomically reclaim an idle previous CPU when possible, otherwise
/// use another idle CPU before comparing runnable load. The previous and waker
/// CPUs only break equal-load ties, preserving locality without letting a
/// centralized wakeup stack independent workers on one run queue.
#[inline]
pub(crate) fn select_wake_run_queue<G: BaseGuard>(task: &AxTaskRef) -> AxRunQueueRef<'static, G> {
    let irq_state = G::acquire();
    #[cfg(not(feature = "smp"))]
    {
        let _ = task;
        AxRunQueueRef {
            inner: unsafe { RUN_QUEUE.current_ref_mut_raw() },
            state: irq_state,
            _phantom: core::marker::PhantomData,
        }
    }
    #[cfg(feature = "smp")]
    {
        let current_cpu = this_cpu_id();
        let last_cpu = task.cpu_id() as usize;
        let cpumask = task.cpumask();
        let idle_start_cpu = NEXT_IDLE_RUN_QUEUE.fetch_add(1, Ordering::Relaxed);
        let fallback_start_cpu = NEXT_RUN_QUEUE.fetch_add(1, Ordering::Relaxed);
        let selection = select_wake_run_queue_index_with(
            cpumask,
            current_cpu,
            last_cpu,
            idle_start_cpu,
            fallback_start_cpu,
            online_cpu_count(),
            run_queue_accepts_tasks,
            try_claim_idle_run_queue,
            run_queue_runnable_load,
        )
        .expect("No initialized run queue matches the task CPU affinity");
        AxRunQueueRef {
            inner: get_run_queue(selection.cpu_id),
            state: irq_state,
            idle_reservation: selection.claimed_idle,
            _phantom: core::marker::PhantomData,
        }
    }
}

/// Selects a run queue for a task migrating because its affinity changed.
///
/// Migration does not claim an idle CPU: the task has already been running,
/// and its `on_cpu` hand-off protocol must remain independent from the fork
/// reservation hint.
#[cfg(feature = "smp")]
#[inline]
fn select_migration_run_queue<G: BaseGuard>(task: &AxTaskRef) -> AxRunQueueRef<'static, G> {
    let irq_state = G::acquire();
    let index = select_run_queue_index(task.cpumask());
    AxRunQueueRef {
        inner: get_run_queue(index),
        state: irq_state,
        idle_reservation: false,
        _phantom: core::marker::PhantomData,
    }
}

/// [`AxRunQueue`] represents a run queue for global system or a specific CPU.
pub(crate) struct AxRunQueue {
    /// The ID of the CPU this run queue is associated with.
    cpu_id: usize,
    /// Number of tasks currently stored in `scheduler`'s ready queue.
    ///
    /// Updates are serialized by `scheduler`; atomic reads let a spawning CPU
    /// compare remote queues without nesting run-queue locks. This is a
    /// placement metric only, so it does not publish task state.
    #[cfg(feature = "smp")]
    ready_tasks: AtomicUsize,
    /// The core scheduler of this run queue.
    /// Since irq and preempt are preserved by the kernel guard hold by `AxRunQueueRef`,
    /// we just use a simple raw spin lock here.
    scheduler: SpinRaw<Scheduler>,
}

/// A reference to the run queue with specific guard.
///
/// Note:
/// [`AxRunQueueRef`] is used to get a reference to the run queue on current CPU
/// or a remote CPU, which is used to add tasks to the run queue or unblock tasks.
/// If you want to perform scheduling operations on the current run queue,
/// see [`CurrentRunQueueRef`].
pub(crate) struct AxRunQueueRef<'a, G: BaseGuard> {
    inner: &'a mut AxRunQueue,
    state: G::State,
    /// Rolls back an uncommitted `IDLE -> RESERVED` placement transition.
    #[cfg(feature = "smp")]
    idle_reservation: bool,
    _phantom: core::marker::PhantomData<G>,
}

impl<G: BaseGuard> Drop for AxRunQueueRef<'_, G> {
    fn drop(&mut self) {
        #[cfg(feature = "smp")]
        if self.idle_reservation {
            release_idle_run_queue_reservation(self.inner.cpu_id);
        }
        G::release(self.state);
    }
}

/// A reference to the current run queue with specific guard.
///
/// Note:
/// [`CurrentRunQueueRef`] is used to get a reference to the run queue on current CPU,
/// in which scheduling operations can be performed.
pub(crate) struct CurrentRunQueueRef<'a, G: BaseGuard> {
    inner: &'a mut AxRunQueue,
    current_task: CurrentTask,
    state: G::State,
    _phantom: core::marker::PhantomData<G>,
}

impl<G: BaseGuard> Drop for CurrentRunQueueRef<'_, G> {
    fn drop(&mut self) {
        G::release(self.state);
    }
}

/// Management operations for run queue, including adding tasks, unblocking tasks, etc.
impl<G: BaseGuard> AxRunQueueRef<'_, G> {
    /// Adds a task to the scheduler.
    ///
    /// This function is used to add a new task to the scheduler.
    pub fn add_task(&mut self, task: AxTaskRef) {
        let cpu_id = self.inner.cpu_id;
        debug!("task add: {} on run_queue {}", task.id_name(), cpu_id);
        assert!(task.is_ready());
        #[cfg(feature = "smp")]
        task.set_cpu_id(cpu_id as _);
        {
            let mut scheduler = self.inner.scheduler.lock();
            scheduler.add_task(task);
            #[cfg(feature = "smp")]
            self.inner.ready_tasks.fetch_add(1, Ordering::Relaxed);
        }
        #[cfg(feature = "smp")]
        {
            // The task is now visible in the scheduler, so the reservation is
            // committed and must be consumed by the target CPU's next switch.
            self.idle_reservation = false;
        }
        #[cfg(all(feature = "smp", feature = "ipi"))]
        kick_remote_cpu(cpu_id);
    }

    /// Unblock one task by inserting it into the run queue.
    ///
    /// This function does nothing if the task is not in [`TaskState::Blocked`],
    /// which means the task is already unblocked by other cores.
    pub fn unblock_task(&mut self, task: AxTaskRef, resched: bool) {
        let task_id_name = if log::log_enabled!(log::Level::Debug) {
            Some(task.id_name())
        } else {
            None
        };
        // Try to change the state of the task from `Blocked` to `Ready`,
        // if successful, the task will be put into this run queue,
        // otherwise, the task is already unblocked by other cores.
        // Note:
        // target task can not be insert into the run queue until it finishes its scheduling process.
        if self
            .inner
            // A wakeup is not a time-slice preemption of the woken task.
            .put_task_with_state(task, TaskState::Blocked, false)
        {
            #[cfg(feature = "smp")]
            {
                // The state CAS and enqueue succeeded; do not roll the target
                // CPU's reservation back when this run-queue reference drops.
                self.idle_reservation = false;
            }
            // Since now, the task to be unblocked is in the `Ready` state.
            let cpu_id = self.inner.cpu_id;
            if let Some(task_id_name) = task_id_name {
                debug!("task unblock: {task_id_name} on run_queue {cpu_id}");
            }
            // Note: when the task is unblocked on another CPU's run queue,
            // we just ignore the `resched` flag.
            if resched && cpu_id == this_cpu_id() {
                #[cfg(feature = "preempt")]
                crate::current().set_preempt_pending(true);
            }
            #[cfg(all(feature = "smp", feature = "ipi"))]
            kick_remote_cpu(cpu_id);
        }
    }
}

/// Core functions of run queue.
impl<G: BaseGuard> CurrentRunQueueRef<'_, G> {
    /// Unblock one task by inserting it into the current CPU's run queue.
    ///
    /// See [`AxRunQueueRef::unblock_task`] for the state-transition details.
    #[cfg(feature = "irq")]
    pub(crate) fn unblock_task(&mut self, task: AxTaskRef, resched: bool) {
        let task_id_name = if log::log_enabled!(log::Level::Debug) {
            Some(task.id_name())
        } else {
            None
        };
        if self
            .inner
            // A wakeup is not a time-slice preemption of the woken task.
            .put_task_with_state(task, TaskState::Blocked, false)
        {
            let cpu_id = self.inner.cpu_id;
            if let Some(task_id_name) = task_id_name {
                debug!("task unblock: {task_id_name} on run_queue {cpu_id}");
            }
            if resched {
                #[cfg(feature = "preempt")]
                crate::current().set_preempt_pending(true);
            }
        }
    }

    #[cfg(feature = "irq")]
    pub fn scheduler_timer_tick(&mut self) {
        let curr = &self.current_task;
        if !curr.is_idle() && self.inner.scheduler.lock().task_tick(curr) {
            #[cfg(feature = "preempt")]
            curr.set_preempt_pending(true);
        }
    }

    /// Yield the current task and reschedule.
    /// This function will put the current task into this run queue with `Ready` state,
    /// and reschedule to the next task on this run queue.
    pub fn yield_current(&mut self) {
        let curr = &self.current_task;
        trace!("task yield: {}", curr.id_name());
        assert!(curr.is_running());

        #[cfg(feature = "smp")]
        if !curr.cpumask().get(self.inner.cpu_id) {
            self.migrate_current_to_affinity();
            return;
        }

        self.inner
            .put_task_with_state(curr.clone(), TaskState::Running, false);

        self.inner.resched();
    }

    /// Migrate the current task to a new run queue matching its CPU affinity and reschedule.
    /// This function will spawn a new `migration_task` to perform the migration, which will set
    /// current task to `Ready` state and select a proper run queue for it according to its CPU affinity,
    /// switch to the migration task immediately after migration task is prepared.
    ///
    /// Note: the ownership of migrating task (which is current task) is handed over to the migration task,
    /// before the migration task inserted it into the target run queue.
    #[cfg(feature = "smp")]
    pub fn migrate_current(&mut self, migration_task: AxTaskRef) {
        let curr = &self.current_task;
        trace!("task migrate: {}", curr.id_name());
        assert!(curr.is_running());

        // Mark current task's state as `Ready`,
        // but, do not put current task to the scheduler of this run queue.
        curr.set_state(TaskState::Ready);

        // Call `switch_to` to reschedule to the migration task that performs the migration directly.
        self.inner.switch_to(crate::current(), migration_task);
    }

    /// Preempts the current task and reschedules.
    /// This function is used to preempt the current task and reschedule
    /// to next task on current run queue.
    ///
    /// This function is called by `current_check_preempt_pending` with IRQs and preemption disabled.
    ///
    /// Note:
    /// preemption may happened in `enable_preempt`, which is called
    /// each time a [`ax_kspin::NoPreemptGuard`] is dropped.
    #[cfg(feature = "preempt")]
    pub fn preempt_resched(&mut self) {
        // There is no need to disable IRQ and preemption here, because
        // they both have been disabled in `current_check_preempt_pending`.
        let curr = &self.current_task;
        assert!(curr.is_running());

        // When we call `preempt_resched()`, both IRQs and preemption must
        // have been disabled by `ax_kernel_guard::NoPreemptIrqSave`. So we need
        // to set `current_disable_count` to 1 in `can_preempt()` to obtain
        // the preemption permission.
        let can_preempt = curr.can_preempt(1);

        trace!(
            "current task is to be preempted: {}, allow={}",
            curr.id_name(),
            can_preempt
        );
        if can_preempt {
            #[cfg(feature = "smp")]
            if !curr.cpumask().get(self.inner.cpu_id) {
                self.migrate_current_to_affinity();
                return;
            }

            self.inner
                .put_task_with_state(curr.clone(), TaskState::Running, true);
            self.inner.resched();
        } else {
            curr.set_preempt_pending(true);
        }
    }

    #[cfg(feature = "preempt")]
    pub fn force_resched(&mut self) {
        self.force_resched_with_preempt_count(1);
    }

    #[cfg(feature = "preempt")]
    fn force_resched_with_preempt_count(&mut self, current_disable_count: usize) {
        let curr = &self.current_task;
        assert!(curr.is_running());

        let can_preempt = curr.can_preempt(current_disable_count);
        trace!(
            "current task is forced to reschedule: {}, allow={}",
            curr.id_name(),
            can_preempt
        );
        if can_preempt {
            #[cfg(feature = "smp")]
            if !curr.cpumask().get(self.inner.cpu_id) {
                self.migrate_current_to_affinity();
                return;
            }

            self.inner
                .put_task_with_state(curr.clone(), TaskState::Running, false);
            self.inner.resched();
        } else {
            curr.set_force_resched_pending(true);
        }
    }

    #[cfg(all(
        feature = "smp",
        feature = "ipi",
        feature = "preempt",
        not(feature = "host-test")
    ))]
    fn force_resched_from_irq() {
        let mut rq = current_run_queue::<ax_kernel_guard::NoOp>();
        rq.force_resched_with_preempt_count(0);
    }

    /// Exit the current task with the specified exit code.
    /// This function will never return.
    pub fn exit_current(&mut self, exit_code: i32) -> ! {
        let curr = &self.current_task;
        debug!("task exit: {}, exit_code={}", curr.id_name(), exit_code);
        assert!(curr.is_running(), "task is not running: {:?}", curr.state());
        assert!(!curr.is_idle());
        if curr.is_init() {
            // Safety: it is called from `current_run_queue::<NoPreemptIrqSave>().exit_current(exit_code)`,
            // which disabled IRQs and preemption.
            unsafe {
                EXITED_TASKS.current_ref_mut_raw().clear();
            }
            ax_hal::power::system_off();
        } else {
            curr.set_state(TaskState::Exited);

            // Notify the joiner task.
            curr.notify_exit(exit_code);

            // Safety: it is called from `current_run_queue::<NoPreemptIrqSave>().exit_current(exit_code)`,
            // which disabled IRQs and preemption.
            unsafe {
                // Push current task to the `EXITED_TASKS` list, which will be consumed by the GC task.
                EXITED_TASKS.current_ref_mut_raw().push_back(curr.clone());
                // Wake up the GC task to drop the exited tasks.
                WAIT_FOR_EXIT.current_ref_mut_raw().notify_one(false);
            }

            // Schedule to next task.
            self.inner.resched();
        }
        unreachable!("task exited!");
    }

    /// Block the current task, put current task into the wait queue and reschedule.
    /// Mark the state of current task as `Blocked`, set the `in_wait_queue` flag as true.
    /// Note:
    ///     1. The caller must hold the lock of the wait queue.
    ///     2. The caller must ensure that the current task is in the running state.
    ///     3. The caller must ensure that the current task is not the idle task.
    ///     4. The lock of the wait queue will be released explicitly after current task is pushed into it.
    pub fn blocked_resched(&mut self, mut wq_guard: WaitQueueGuard) {
        let curr = &self.current_task;
        assert!(curr.is_running());
        assert!(!curr.is_idle());
        // we must not block current task with preemption disabled.
        // Current expected preempt count is 2.
        // 1 for `NoPreemptIrqSave`, 1 for wait queue's `SpinNoIrq`.
        #[cfg(feature = "preempt")]
        assert!(curr.can_preempt(2));

        // Mark the task as blocked, this has to be done before adding it to the wait queue
        // while holding the lock of the wait queue.
        curr.set_state(TaskState::Blocked);

        // A preemptive future wake can re-enter a wait path before a previous
        // wait-queue entry has been consumed. Avoid leaving a stale duplicate
        // waiter that may receive mutex ownership after the task is running.
        if !curr.in_wait_queue() {
            curr.set_in_wait_queue(true);
            wq_guard.push_back(curr.clone());
        }
        // Drop the lock of wait queue explicitly.
        drop(wq_guard);

        // Current task's state has been changed to `Blocked` and added to the wait queue.
        // Note that the state may have been set as `Ready` in `unblock_task()`,
        // see `unblock_task()` for details.

        debug!("task block: {}", curr.id_name());
        self.inner.resched();
    }

    /// Block the current task, put current task into the wait queue and reschedule.
    /// This is special just for future.
    pub fn future_blocked_resched(&mut self, mut woke: SpinNoIrqGuard<'_, bool>) {
        let curr = &self.current_task;
        assert!(curr.is_running());
        assert!(!curr.is_idle());
        // we must not block current task with preemption disabled.
        // Current expected preempt count is 2 for `NoPreemptIrqSave` and `woke`.
        #[cfg(feature = "preempt")]
        assert!(curr.can_preempt(2));

        // Mark the task as blocked, this has to be done before adding it to the wait queue
        // while holding the lock of the wait queue.
        curr.set_state(TaskState::Blocked);
        *woke = false;
        drop(woke);

        // Current task's state has been changed to `Blocked` and added to the wait queue.
        // Note that the state may have been set as `Ready` in `unblock_task()`,
        // see `unblock_task()` for details.

        debug!("task block: {}", curr.id_name());
        self.inner.resched();
    }

    #[cfg(feature = "irq")]
    pub fn sleep_until(&mut self, deadline: ax_hal::time::TimeValue) {
        let curr = &self.current_task;
        debug!("task sleep: {}, deadline={:?}", curr.id_name(), deadline);
        assert!(curr.is_running());
        assert!(!curr.is_idle());

        while ax_hal::time::monotonic_time() < deadline {
            crate::timers::set_alarm_wakeup(deadline, curr.clone());
            curr.set_state(TaskState::Blocked);
            self.inner.resched();
        }
    }

    pub fn set_current_priority(&mut self, prio: isize) -> bool {
        self.inner
            .scheduler
            .lock()
            .set_priority(&self.current_task, prio)
    }

    #[cfg(feature = "smp")]
    fn migrate_current_to_affinity(&mut self) {
        let curr = self.current_task.clone();
        let migration_task = TaskInner::new(
            move || crate::run_queue::migrate_entry(curr),
            "migration-task".into(),
            crate::default_task_stack_size(),
        )
        .into_arc();

        self.migrate_current(migration_task);
    }
}

impl AxRunQueue {
    /// Create a new run queue for the specified CPU.
    /// The run queue is initialized with a per-CPU gc task in its scheduler.
    fn new(cpu_id: usize) -> Self {
        let gc_task =
            TaskInner::new(gc_entry, "gc".into(), crate::default_task_stack_size()).into_arc();
        // gc task should be pinned to the current CPU.
        gc_task.set_cpumask(AxCpuMask::one_shot(cpu_id));

        let mut scheduler = Scheduler::new();
        scheduler.add_task(gc_task);
        Self {
            cpu_id,
            #[cfg(feature = "smp")]
            ready_tasks: AtomicUsize::new(1),
            scheduler: SpinRaw::new(scheduler),
        }
    }

    /// Puts target task into current run queue with `Ready` state
    /// if its state matches `current_state` (except idle task).
    ///
    /// If `preempt`, keep current task's time slice, otherwise reset it.
    ///
    /// Returns `true` if the target task is put into this run queue successfully,
    /// otherwise `false`.
    fn put_task_with_state(
        &mut self,
        task: AxTaskRef,
        current_state: TaskState,
        preempt: bool,
    ) -> bool {
        // If the task's state matches `current_state`, set its state to `Ready` and
        // put it back to the run queue (except idle task).
        if task.transition_state(current_state, TaskState::Ready) && !task.is_idle() {
            #[cfg(feature = "smp")]
            let waking_current_task = current_state == TaskState::Blocked
                && self.cpu_id == this_cpu_id()
                && crate::current().ptr_eq(&task);
            // A blocked task woken here may still be finishing its context
            // switch-out on its owning CPU: `on_cpu == true` means its registers
            // are not yet fully saved. It must NOT be made runnable (enqueued)
            // until `on_cpu` is false, or another CPU could resume it with stale
            // registers. Pairs with `clear_prev_task_on_cpu()`.
            //
            // We must NOT busy-spin on `on_cpu` for a task owned by a *remote*
            // CPU (as the old code did): two CPUs each spinning with IRQs off,
            // each waiting for the other to reach `clear_prev_task_on_cpu()`, is
            // a mutual deadlock (whole-board freeze). Instead, hand the enqueue
            // to the owning CPU via a lock-free stash it drains once its context
            // is saved. `waking_current_task` (self-wake on this CPU, mid-switch)
            // keeps the old inline behavior: this CPU finishes the switch in
            // program order when it returns.
            #[cfg(feature = "smp")]
            if current_state == TaskState::Blocked && !waking_current_task && task.on_cpu() {
                // Record where the task must land, then stash a reference for the
                // owning CPU to enqueue from `clear_prev_task_on_cpu()`.
                task.set_cpu_id(self.cpu_id as _);
                task.stash_wake(task.clone());
                // Re-check under the SeqCst handshake. If still on its owning CPU,
                // that CPU drains the stash after its switch completes — done.
                if task.on_cpu() {
                    return false;
                }
                // `on_cpu` cleared meanwhile: the owning CPU may already have
                // passed its drain point. Whichever side wins `take_wake` does
                // the enqueue exactly once.
                if task.take_wake().is_none() {
                    // Owner won the swap; it enqueues + kicks the target.
                    return false;
                }
                // We won: the reclaimed reference is dropped here; fall through
                // and enqueue our own `task` (its context is now saved).
            }
            // TODO: priority
            #[cfg(feature = "smp")]
            task.set_cpu_id(self.cpu_id as _);
            {
                let mut scheduler = self.scheduler.lock();
                scheduler.put_prev_task(task, preempt);
                #[cfg(feature = "smp")]
                self.ready_tasks.fetch_add(1, Ordering::Relaxed);
            }
            true
        } else {
            false
        }
    }

    /// Core reschedule subroutine.
    /// Pick the next task to run and switch to it.
    fn resched(&mut self) {
        let next = {
            let mut scheduler = self.scheduler.lock();
            let next = scheduler.pick_next_task();
            #[cfg(feature = "smp")]
            if next.is_some() {
                let previous = self.ready_tasks.fetch_sub(1, Ordering::Relaxed);
                debug_assert!(previous > 0, "ready task count must not underflow");
            }
            next
        }
        .unwrap_or_else(|| unsafe {
            // Safety: IRQs must be disabled at this time.
            IDLE_TASK.current_ref_raw().get_unchecked().clone()
        });
        assert!(
            next.is_ready(),
            "next {} is not ready: {:?}",
            next.id_name(),
            next.state()
        );
        self.switch_to(crate::current(), next);
    }

    fn switch_to(&mut self, prev_task: CurrentTask, next_task: AxTaskRef) {
        // Make sure that IRQs are disabled by kernel guard or other means.
        #[cfg(all(feature = "irq", not(feature = "host-test")))]
        assert!(
            !ax_hal::asm::irqs_enabled(),
            "IRQs must be disabled during scheduling"
        );
        trace!(
            "context switch: {} -> {}",
            prev_task.id_name(),
            next_task.id_name()
        );
        prev_task.check_stack_canary();
        #[cfg(feature = "preempt")]
        next_task.set_preempt_pending(false);
        next_task.set_state(TaskState::Running);

        #[cfg(feature = "smp")]
        if next_task.is_idle() {
            // Do not overwrite a reservation made while this CPU was idle.
            // A failed enqueue rolls RESERVED back through its RAII token;
            // a committed enqueue is consumed when the CPU switches to it.
            let _ = RUN_QUEUE_ACTIVITY[this_cpu_id()].compare_exchange(
                RUN_QUEUE_ACTIVITY_BUSY,
                RUN_QUEUE_ACTIVITY_IDLE,
                Ordering::Release,
                Ordering::Relaxed,
            );
        } else {
            RUN_QUEUE_ACTIVITY[this_cpu_id()].store(RUN_QUEUE_ACTIVITY_BUSY, Ordering::Release);
        }
        if prev_task.ptr_eq(&next_task) {
            return;
        }

        // Claim the task as running, we do this before switching to it
        // such that any running task will have this set.
        #[cfg(feature = "smp")]
        next_task.set_on_cpu(true);

        #[cfg(feature = "task-ext")]
        {
            use crate::TaskExt;

            if let Some(ext) = prev_task.task_ext() {
                ext.on_leave()
            }
            if let Some(ext) = next_task.task_ext() {
                ext.on_enter()
            }
        }

        // `prev_task.state()` must be sampled before the architectural switch:
        // callers like `exit_current` already set it to `Exited`/`Blocked`,
        // and that pre-switch state is what `sched:sched_switch` reports.
        #[cfg(feature = "tracepoint-hooks")]
        ax_crate_interface::call_interface!(
            crate::sched_tracepoint::SchedTracepoint::on_sched_switch(
                prev_task.id().as_u64(),
                next_task.id().as_u64(),
                prev_task.state() as u32,
            )
        );

        unsafe {
            let prev_ctx_ptr = prev_task.ctx_mut_ptr();
            let next_ctx_ptr = next_task.ctx_mut_ptr();

            // Store a raw pointer to prev_task in PREV_TASK.
            // Safety: prev_task is alive (Arc held on caller's stack) and will
            // remain so through clear_prev_task_on_cpu() below.
            #[cfg(feature = "smp")]
            {
                *PREV_TASK.current_ref_mut_raw() =
                    Some(NonNull::new(Arc::as_ptr(&prev_task) as *mut _).unwrap());
            }

            // The strong reference count of `prev_task` will be decremented by 1,
            // but won't be dropped until `gc_entry()` is called.
            assert!(Arc::strong_count(&prev_task) > 1);
            assert!(Arc::strong_count(&next_task) >= 1);

            CurrentTask::set_current(prev_task, next_task);

            (*prev_ctx_ptr).switch_to(&*next_ctx_ptr);

            // The current task is now **next_task** on this CPU, so clear `prev_task.on_cpu`
            // to indicate that it has finished its scheduling process and no longer running on this CPU.
            #[cfg(feature = "smp")]
            clear_prev_task_on_cpu();
        }
    }
}

fn gc_entry() {
    loop {
        // Drop all exited tasks and recycle resources.
        let n = EXITED_TASKS.with_current(|exited_tasks| exited_tasks.len());
        for _ in 0..n {
            // Do not do the slow drops in the critical section.
            let task = EXITED_TASKS.with_current(|exited_tasks| exited_tasks.pop_front());
            if let Some(task) = task {
                if Arc::strong_count(&task) == 1 {
                    // If I'm the last holder of the task, drop it immediately.
                    drop(task);
                } else {
                    // Otherwise (e.g, `switch_to` is not completed, held by the
                    // joiner, etc), push it back and wait for them to drop first.
                    EXITED_TASKS.with_current(|exited_tasks| exited_tasks.push_back(task));
                }
            }
        }
        // Always wait with a timeout to:
        // 1. Yield CPU to allow other tasks to complete `switch_to` and drop references
        // 2. Handle the race condition where `notify_one` is called before the GC task enters wait,
        //    causing the notification to be lost.
        // Note: we cannot block current task with preemption disabled,
        // use `current_ref_raw` to get the `WAIT_FOR_EXIT`'s reference here to avoid the use of `NoPreemptGuard`.
        // Since gc task is pinned to the current CPU, there is no effect if the gc task is preempted during the process.
        #[cfg(feature = "irq")]
        unsafe {
            let _timeout = WAIT_FOR_EXIT
                .current_ref_raw()
                .wait_timeout(core::time::Duration::from_millis(100));
        }
        #[cfg(not(feature = "irq"))]
        unsafe {
            WAIT_FOR_EXIT.current_ref_raw().wait();
        }
    }
}

/// The task routine for migrating the current task to the correct CPU.
///
/// It calls `select_migration_run_queue` to get the correct run queue for the task, and
/// then puts the task to the scheduler of target run queue.
#[cfg(feature = "smp")]
pub(crate) fn migrate_entry(migrated_task: AxTaskRef) {
    let rq = select_migration_run_queue::<ax_kernel_guard::NoPreemptIrqSave>(&migrated_task);
    let cpu_id = rq.inner.cpu_id;
    migrated_task.set_cpu_id(cpu_id as _);
    {
        let mut scheduler = rq.inner.scheduler.lock();
        scheduler.put_prev_task(migrated_task, false);
        rq.inner.ready_tasks.fetch_add(1, Ordering::Relaxed);
    }
    #[cfg(all(feature = "smp", feature = "ipi"))]
    // Current-task migration cannot make progress until the target CPU runs
    // the migrated task, so do not let a stale coalescing bit suppress this IPI.
    force_kick_remote_cpu(cpu_id);
}

/// Clear the `on_cpu` field of the previous task running on this CPU, then
/// complete any cross-core wake that was deferred while it was still `on_cpu`.
#[cfg(feature = "smp")]
pub(crate) unsafe fn clear_prev_task_on_cpu() {
    let prev = unsafe { PREV_TASK.current_ref_mut_raw() }
        .take()
        .expect("PREV_TASK should have been set by switch_to");
    // Safety: prev_task's Arc is still alive on the caller's stack at this point
    // (switch_to has not yet returned), so the pointer is valid.
    let prev = unsafe { prev.as_ref() };
    // Publish that the context is fully saved. The SeqCst store pairs with the
    // waker's `on_cpu()`/`take_wake()` handshake in `put_task_with_state`.
    prev.set_on_cpu(false);
    // Drain a wake that raced our switch-out. `take_wake` is the single arbiter:
    // if the waker did not reclaim it (it saw `on_cpu` still true), we get the
    // owned reference and enqueue it now that the context is saved.
    if let Some(task) = prev.take_wake() {
        let target = task.cpu_id() as usize;
        // Leaf lock: `resched()` already dropped this CPU's scheduler lock before
        // `switch_to`, so this takes only the target run queue's lock.
        get_run_queue(target)
            .scheduler
            .lock()
            .put_prev_task(task, false);
        if target != this_cpu_id() {
            // Remote target: ask that CPU to reschedule so it picks the task up
            // (and wakes if it is idle in `wait_for_irqs`).
            #[cfg(feature = "ipi")]
            kick_remote_cpu(target);
        } else {
            // Local target: `kick_remote_cpu(self)` is a no-op, so the reschedule
            // the remote waker's IPI used to deliver here would be lost — the
            // task could sit un-run until the next tick, or indefinitely if this
            // CPU just switched to `idle` and is about to `wait_for_irqs()`.
            // `target == this_cpu_id()` arises when `select_wake_run_queue()`
            // falls back to the task's `last_cpu`, which is this owning CPU.
            // Request a reschedule on THIS CPU instead: the current task
            // (`next_task`, possibly `idle`) is forced to reschedule when the
            // switch chain unwinds and its preempt guard is released
            // (`current_check_preempt_pending` consumes the flag), mirroring the
            // reschedule the IPI path (`request_current_reschedule`) performed.
            #[cfg(feature = "preempt")]
            crate::current().set_force_resched_pending(true);
        }
    }
}
pub(crate) fn init() {
    let cpu_id = this_cpu_id();

    // Create the `idle` task (not current task).
    // The idle task will run when there is no other runnable task.
    #[cfg(feature = "lockdep")]
    let idle_task_stack_size = crate::default_task_stack_size();
    // TODO: Consider unifying the non-lockdep idle stack size with the task stack configuration.
    #[cfg(not(feature = "lockdep"))]
    let idle_task_stack_size = 16384;
    let idle_task = TaskInner::new(|| crate::run_idle(), "idle".into(), idle_task_stack_size);
    // idle task should be pinned to the current CPU.
    idle_task.set_cpumask(AxCpuMask::one_shot(cpu_id));
    IDLE_TASK.with_current(|i| {
        i.init_once(idle_task.into_arc());
    });

    // Put the subsequent execution into the `main` task.
    let main_task = TaskInner::new_init("main".into(), main_task_stack()).into_arc();
    main_task.set_state(TaskState::Running);
    unsafe { CurrentTask::init_current(main_task) }

    RUN_QUEUE.with_current(|rq| {
        rq.init_once(AxRunQueue::new(cpu_id));
    });
    unsafe {
        RUN_QUEUES[cpu_id].write(RUN_QUEUE.current_ref_mut_raw());
    }
    #[cfg(feature = "smp")]
    RUN_QUEUE_INITIALIZED[cpu_id].store(true, Ordering::Release);
}

pub(crate) fn init_secondary(stack_ptr: VirtAddr, stack_size: usize) {
    let cpu_id = this_cpu_id();

    // Put the subsequent execution into the `idle` task.
    let idle_task = TaskInner::new_init(
        "idle".into(),
        TaskStack::borrowed(stack_ptr, stack_size, TASK_STACK_ALIGN),
    )
    .into_arc();
    idle_task.set_state(TaskState::Running);
    IDLE_TASK.with_current(|i| {
        i.init_once(idle_task.clone());
    });
    unsafe { CurrentTask::init_current(idle_task) }

    RUN_QUEUE.with_current(|rq| {
        rq.init_once(AxRunQueue::new(cpu_id));
    });
    unsafe {
        RUN_QUEUES[cpu_id].write(RUN_QUEUE.current_ref_mut_raw());
    }
    #[cfg(feature = "smp")]
    {
        RUN_QUEUE_ACTIVITY[cpu_id].store(RUN_QUEUE_ACTIVITY_IDLE, Ordering::Relaxed);
        RUN_QUEUE_INITIALIZED[cpu_id].store(true, Ordering::Release);
    }
}
