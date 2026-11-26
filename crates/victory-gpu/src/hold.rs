//! Work a queue holds back (decision 0007, "Queues wait on memory values").
//!
//! The console's lists wait on memory values (`Wait32BitValueX`,
//! `Wait64BitValueX`) that lists on other queues write at their end
//! (`Write32/64BitValueBottomOfPipeX`): Grand Theft Auto VI's four compute
//! queues start each list with a wait for the frame's value of a word that
//! a graphics list writes, and some lists wait again after their first
//! commands. Our submissions run to completion when the title makes them,
//! so a queue here is a FIFO of what the title submitted to it: segments of
//! lists (a list is cut where it waits after commands, `crate::list`), its
//! Signals and its fence Waits, in their order. The head of a queue runs
//! once its waits hold; consecutive heads that may run go to the host GPU
//! together (a piece); after every piece the heads of all queues are tested
//! again until none can move. Order within a queue never changes, only the
//! interleaving across queues.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// A wait for a memory value, as a list records it: the value at `address`
/// (`bytes` 4 or 8, the 32-bit form masked) compared with `reference` by a
/// D3D12 comparison function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ValueWait {
    pub(crate) address: u64,
    pub(crate) bytes: u8,
    pub(crate) comparison: u32,
    pub(crate) reference: u64,
    pub(crate) mask: u64,
}

/// D3D12 comparison functions (`D3D12_COMPARISON_FUNC`).
const NEVER: u32 = 1;
const LESS: u32 = 2;
const EQUAL: u32 = 3;
const LESS_EQUAL: u32 = 4;
const GREATER: u32 = 5;
const NOT_EQUAL: u32 = 6;
const GREATER_EQUAL: u32 = 7;
const ALWAYS: u32 = 8;

impl ValueWait {
    /// Whether it holds with `read` giving the memory's value: `None` for a
    /// comparison we do not know (counted by the caller, taken as holding)
    /// and for memory that cannot be read (taken as not holding).
    pub(crate) fn holds(&self, read: &dyn Fn(u64, u8) -> Option<u64>) -> Option<bool> {
        if !(NEVER..=ALWAYS).contains(&self.comparison) {
            return None;
        }
        let value = read(self.address, self.bytes)? & self.mask;
        let reference = self.reference;
        Some(match self.comparison {
            NEVER => false,
            LESS => value < reference,
            EQUAL => value == reference,
            LESS_EQUAL => value <= reference,
            GREATER => value > reference,
            NOT_EQUAL => value != reference,
            GREATER_EQUAL => value >= reference,
            _ => true,
        })
    }
}

/// What a queue holds, in its order.
pub(crate) enum Item<S> {
    /// A list segment and the waits at its start.
    Segment(S, Vec<ValueWait>),
    /// A fence Signal: the fence and its value.
    Signal(usize, u64),
    /// A fence Wait: until the fence reaches the value.
    FenceWait(usize, u64),
}

/// What a pump did, for the frame's accounting.
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct Pumped {
    /// Pieces given to the host GPU, and the segments in them.
    pub(crate) pieces: u32,
    pub(crate) segments: u32,
    /// Of those, segments that had waited (not run when submitted): their
    /// queue and how long.
    pub(crate) waited: Vec<(usize, Duration)>,
    /// Waits with a comparison we do not know (taken as holding).
    pub(crate) unknown: u32,
}

/// Predicate checking whether a ready segment can join an in-progress
/// coalesced batch.
#[cfg(any(windows, test))]
pub(crate) type CanJoin<'a, S> = &'a dyn Fn(&[(S, bool)], &S) -> bool;

/// One ready step popped from [`Held`] so the caller can release the `HELD`
/// lock while running it on the host GPU.
#[cfg(any(windows, test))]
pub(crate) enum Ready<S> {
    /// Ready segments popped as one host submission (from one queue, or
    /// coalesced across queues and bottom-of-pipe dependencies when
    /// `coalesce` is enabled), paired with whether a global UAV barrier must
    /// precede each segment, plus any mid-batch `Item::Signal`s to attach
    /// behind the submission. Every queue in `queues` stays marked running
    /// until [`Held::finish_running`] is called so later submissions and
    /// signals on those queues wait behind it.
    Segments {
        queues: Vec<usize>,
        segments: Vec<(S, bool)>,
        signals: Vec<(usize, u64)>,
        waited: Vec<(usize, Duration)>,
        unknown: u32,
        coalesced_bop_waits: u32,
        hazard_splits: u32,
    },
    /// A fence signal ready at the head of a queue with no preceding segments.
    Signal { fence: usize, value: u64 },
}

/// Every queue's held work.
pub(crate) struct Held<S> {
    queues: Vec<(usize, Fifo<S>)>,
    /// Queues whose popped segments are currently launching outside the lock.
    running: Vec<usize>,
}

/// A queue's items, each with when it was pushed and whether it is fresh
/// (not yet found waiting).
type Fifo<S> = VecDeque<(Item<S>, Instant, bool)>;

impl<S> Default for Held<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Held<S> {
    pub(crate) const fn new() -> Self {
        Held {
            queues: Vec::new(),
            running: Vec::new(),
        }
    }
}

/// What the pump needs from the outside world.
pub(crate) struct World<'a, S> {
    /// A memory value (address, bytes).
    pub(crate) read: &'a dyn Fn(u64, u8) -> Option<u64>,
    /// A fence's value.
    pub(crate) fence: &'a dyn Fn(usize) -> u64,
    /// Runs a piece (a queue, its segments in order) to completion, the
    /// bottom-of-pipe writes included.
    pub(crate) run: &'a mut dyn FnMut(usize, Vec<S>),
    /// Signals a fence.
    pub(crate) signal: &'a mut dyn FnMut(usize, u64),
}

impl<S> Held<S> {
    /// Appends `item` to `queue`'s FIFO, marked fresh: whether it waited is
    /// decided by its first test.
    pub(crate) fn push(&mut self, queue: usize, item: Item<S>) {
        let now = Instant::now();
        match self.queues.iter_mut().find(|(q, _)| *q == queue) {
            Some((_, items)) => items.push_back((item, now, true)),
            None => self.queues.push((queue, VecDeque::from([(item, now, true)]))),
        }
    }

    /// Runs every queue's heads that may run, until none can.
    pub(crate) fn pump(&mut self, w: &mut World<S>) -> Pumped {
        let mut done = Pumped::default();
        loop {
            let mut moved = false;
            for (queue, items) in &mut self.queues {
                let mut piece = Vec::new();
                loop {
                    let ready = match items.front_mut() {
                        None => false,
                        Some((Item::Segment(_, waits), since, fresh)) => {
                            let mut ok = true;
                            for wait in waits.iter() {
                                match wait.holds(w.read) {
                                    Some(true) => {}
                                    Some(false) => ok = false,
                                    None => done.unknown += 1,
                                }
                            }
                            if !ok {
                                *fresh = false;
                            } else if !*fresh {
                                done.waited.push((*queue, since.elapsed()));
                            }
                            ok
                        }
                        // A Signal or Wait keeps its place: the piece before
                        // it runs first.
                        Some((Item::Signal(..), ..)) => {
                            if !piece.is_empty() {
                                (w.run)(*queue, std::mem::take(&mut piece));
                                done.pieces += 1;
                            }
                            true
                        }
                        Some((Item::FenceWait(fence, value), _, fresh)) => {
                            let ok = (w.fence)(*fence) >= *value;
                            if !ok {
                                *fresh = false;
                            }
                            ok
                        }
                    };
                    if !ready {
                        break;
                    }
                    moved = true;
                    match items.pop_front().map(|(item, ..)| item) {
                        Some(Item::Segment(s, _)) => {
                            piece.push(s);
                            done.segments += 1;
                        }
                        Some(Item::Signal(fence, value)) => (w.signal)(fence, value),
                        Some(Item::FenceWait(..)) | None => {}
                    }
                }
                if !piece.is_empty() {
                    (w.run)(*queue, piece);
                    done.pieces += 1;
                }
            }
            if !moved {
                break;
            }
        }
        self.queues.retain(|(_, items)| !items.is_empty());
        done
    }

    /// The queues still holding work: (queue, items, how long the head
    /// waited, the head's waits that do not hold).
    pub(crate) fn blocked(&self, read: &dyn Fn(u64, u8) -> Option<u64>) -> Vec<(usize, usize, Duration, Vec<ValueWait>)> {
        self.queues
            .iter()
            .filter_map(|(q, items)| {
                let (head, since, _) = items.front()?;
                let unmet = match head {
                    Item::Segment(_, waits) => waits.iter().filter(|w| w.holds(read) == Some(false)).copied().collect(),
                    _ => Vec::new(),
                };
                Some((*q, items.len(), since.elapsed(), unmet))
            })
            .collect()
    }

    /// Pops the next ready batch of segments or ready fence signal so the
    /// caller can drop the `HELD` lock before invoking the host GPU.
    /// When `coalesce` is `true`, accumulates ready segments across all
    /// non-running queues (including segments whose memory waits are satisfied
    /// by an earlier segment's deferred bottom-of-pipe writes via `will_write`,
    /// gated at cross-queue / BoP / fence-wait boundaries by `can_join`) and
    /// defers mid-batch `Item::Signal`s into `Ready::Segments::signals`.
    /// When [`Ready::Segments`] is returned, the caller must call
    /// [`Self::finish_running`] with `queues` once the submission call returns.
    #[cfg(any(windows, test))]
    pub(crate) fn take_ready_with(
        &mut self,
        read: &dyn Fn(u64, u8) -> Option<u64>,
        fence: &dyn Fn(usize) -> u64,
        will_write: &dyn Fn(&S, &ValueWait) -> bool,
        can_join: CanJoin<'_, S>,
        coalesce: bool,
    ) -> Option<Ready<S>> {
        if !coalesce {
            return self.take_ready_per_queue(read, fence);
        }
        let mut piece: Vec<(S, bool)> = Vec::new();
        let mut signals: Vec<(usize, u64)> = Vec::new();
        let mut queues_touched: Vec<usize> = Vec::new();
        let mut waited: Vec<(usize, Duration)> = Vec::new();
        let mut unknown = 0u32;
        let mut coalesced_bop_waits = 0u32;
        let mut hazard_splits = 0u32;
        let mut last_queue: Option<usize> = None;
        let mut split_now = false;

        let mark_touched = |touched: &mut Vec<usize>, q: usize| {
            if !touched.contains(&q) {
                touched.push(q);
            }
        };

        loop {
            let mut moved = false;
            for (queue, items) in &mut self.queues {
                if self.running.contains(queue) {
                    continue;
                }
                let mut pending_fence_barrier = false;
                let mut popped_pending_waits: Vec<(Item<S>, Instant, bool)> = Vec::new();
                loop {
                    match items.front_mut() {
                        None => break,
                        Some((Item::FenceWait(f, v), _, fresh)) => {
                            let mem_ok = fence(*f) >= *v;
                            let pending_ok = !mem_ok
                                && signals.iter().any(|&(sf, sv)| sf == *f && sv >= *v);
                            if !mem_ok && !pending_ok {
                                *fresh = false;
                                break;
                            }
                            if let Some(entry) = items.pop_front() {
                                if pending_ok {
                                    pending_fence_barrier = true;
                                    popped_pending_waits.push(entry);
                                }
                                moved = true;
                            }
                        }
                        Some((Item::Signal(..), ..)) => {
                            if piece.is_empty() && signals.is_empty() {
                                if let Some((Item::Signal(f, v), ..)) = items.pop_front() {
                                    self.queues.retain(|(_, q)| !q.is_empty());
                                    return Some(Ready::Signal { fence: f, value: v });
                                }
                                break;
                            }
                            if let Some((Item::Signal(f, v), ..)) = items.pop_front() {
                                popped_pending_waits.clear();
                                signals.push((f, v));
                                mark_touched(&mut queues_touched, *queue);
                                moved = true;
                            }
                        }
                        Some((Item::Segment(seg, waits), since, fresh)) => {
                            let mut ok = true;
                            let mut via_bop = 0u32;
                            let mut seg_unknown = 0u32;
                            for wait in waits.iter() {
                                match wait.holds(read) {
                                    Some(true) => {}
                                    Some(false) => {
                                        if piece
                                            .iter()
                                            .rev()
                                            .any(|(prev, _)| will_write(prev, wait))
                                        {
                                            via_bop += 1;
                                        } else {
                                            ok = false;
                                        }
                                    }
                                    None => seg_unknown += 1,
                                }
                            }
                            let is_join_boundary = !piece.is_empty()
                                && (last_queue != Some(*queue)
                                    || via_bop > 0
                                    || pending_fence_barrier);
                            if !ok {
                                *fresh = false;
                                unknown += seg_unknown;
                                for entry in popped_pending_waits.drain(..).rev() {
                                    items.push_front(entry);
                                }
                                break;
                            }
                            if is_join_boundary && !can_join(&piece, seg) {
                                hazard_splits += 1;
                                split_now = true;
                                for entry in popped_pending_waits.drain(..).rev() {
                                    items.push_front(entry);
                                }
                                break;
                            }
                            if !*fresh {
                                waited.push((*queue, since.elapsed()));
                            }
                            unknown += seg_unknown;
                            coalesced_bop_waits += via_bop;
                            let barrier_before =
                                via_bop > 0 || std::mem::take(&mut pending_fence_barrier);
                            popped_pending_waits.clear();
                            if let Some((Item::Segment(s, _), ..)) = items.pop_front() {
                                piece.push((s, barrier_before));
                                last_queue = Some(*queue);
                                mark_touched(&mut queues_touched, *queue);
                                moved = true;
                            }
                        }
                    }
                }
                for entry in popped_pending_waits.drain(..).rev() {
                    items.push_front(entry);
                }
                if split_now {
                    break;
                }
            }
            if split_now || !moved {
                break;
            }
        }
        self.queues.retain(|(_, items)| !items.is_empty());
        if !piece.is_empty() {
            for &q in &queues_touched {
                if !self.running.contains(&q) {
                    self.running.push(q);
                }
            }
            return Some(Ready::Segments {
                queues: queues_touched,
                segments: piece,
                signals,
                waited,
                unknown,
                coalesced_bop_waits,
                hazard_splits,
            });
        }
        None
    }

    #[cfg(any(windows, test))]
    fn take_ready_per_queue(
        &mut self,
        read: &dyn Fn(u64, u8) -> Option<u64>,
        fence: &dyn Fn(usize) -> u64,
    ) -> Option<Ready<S>> {
        for (queue, items) in &mut self.queues {
            if self.running.contains(queue) {
                continue;
            }
            let mut segments = Vec::new();
            let mut waited = Vec::new();
            let mut unknown = 0u32;
            loop {
                match items.front_mut() {
                    None => break,
                    Some((Item::FenceWait(f, v), _, fresh)) => {
                        if fence(*f) >= *v {
                            items.pop_front();
                            continue;
                        }
                        *fresh = false;
                        break;
                    }
                    Some((Item::Signal(..), ..)) => {
                        if !segments.is_empty() {
                            break;
                        }
                        if let Some((Item::Signal(f, v), ..)) = items.pop_front() {
                            self.queues.retain(|(_, q)| !q.is_empty());
                            return Some(Ready::Signal { fence: f, value: v });
                        }
                        break;
                    }
                    Some((Item::Segment(_, waits), since, fresh)) => {
                        let mut ok = true;
                        for wait in waits.iter() {
                            match wait.holds(read) {
                                Some(true) => {}
                                Some(false) => ok = false,
                                None => unknown += 1,
                            }
                        }
                        if !ok {
                            *fresh = false;
                            break;
                        }
                        if !*fresh {
                            waited.push((*queue, since.elapsed()));
                        }
                        if let Some((Item::Segment(s, _), ..)) = items.pop_front() {
                            segments.push((s, false));
                        }
                    }
                }
            }
            if !segments.is_empty() {
                let q = *queue;
                self.running.push(q);
                return Some(Ready::Segments {
                    queues: vec![q],
                    segments,
                    signals: Vec::new(),
                    waited,
                    unknown,
                    coalesced_bop_waits: 0,
                    hazard_splits: 0,
                });
            }
        }
        self.queues.retain(|(_, items)| !items.is_empty());
        None
    }

    /// Clears the running markers set by [`Self::take_ready_with`] once the
    /// popped batch has been handed to the host GPU.
    #[cfg(any(windows, test))]
    pub(crate) fn finish_running(&mut self, queues: &[usize]) {
        self.running.retain(|q| !queues.contains(q));
        self.queues.retain(|(_, items)| !items.is_empty());
    }

    /// Whether no queue holds anything and none is currently running.
    pub(crate) fn is_empty(&self) -> bool {
        self.running.is_empty() && self.queues.is_empty()
    }

    /// Whether `queue` holds anything or has a popped batch still launching.
    pub(crate) fn holds_work(&self, queue: usize) -> bool {
        self.running.contains(&queue)
            || self.queues.iter().any(|(q, items)| *q == queue && !items.is_empty())
    }

    /// Releases work held longer than `limit` at the heads of blocked
    /// queues, as if its waits held: the count of segments forced.
    // shortcut: a safety valve against waits nothing will satisfy (a write
    // we do not emulate), which would hold the title's fences back until
    // its 30 s watchdog ends the run; remove once runs show it never fires.
    pub(crate) fn force_older_than(&mut self, limit: Duration) -> Vec<(usize, Vec<ValueWait>)> {
        let mut forced = Vec::new();
        for (q, items) in &mut self.queues {
            if let Some((Item::Segment(_, waits), since, _)) = items.front_mut()
                && since.elapsed() >= limit
            {
                forced.push((*q, std::mem::take(waits)));
            }
        }
        forced
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// Memory, fences and a log of what ran, for a recording executor.
    #[derive(Default)]
    struct Fake {
        memory: HashMap<u64, u64>,
        fences: HashMap<usize, u64>,
        log: Vec<String>,
        /// Bottom-of-pipe writes of each segment: (address, value).
        writes: HashMap<&'static str, Vec<(u64, u64)>>,
    }

    fn ge(address: u64, reference: u64) -> ValueWait {
        ValueWait { address, bytes: 8, comparison: GREATER_EQUAL, reference, mask: u64::MAX }
    }

    fn pump(held: &mut Held<&'static str>, fake: &RefCell<Fake>) -> Pumped {
        let read = |a: u64, _| Some(*fake.borrow().memory.get(&a).unwrap_or(&0));
        let fence = |f: usize| *fake.borrow().fences.get(&f).unwrap_or(&0);
        let mut run = |q: usize, segments: Vec<&'static str>| {
            let mut f = fake.borrow_mut();
            for s in &segments {
                for (a, v) in f.writes.get(s).cloned().unwrap_or_default() {
                    f.memory.insert(a, v);
                }
            }
            f.log.push(format!("q{q}: {}", segments.join(" ")));
        };
        let mut signal = |fence: usize, value: u64| {
            let mut f = fake.borrow_mut();
            f.fences.insert(fence, value);
            f.log.push(format!("signal {fence} {value}"));
        };
        held.pump(&mut World { read: &read, fence: &fence, run: &mut run, signal: &mut signal })
    }

    #[test]
    fn comparisons_test_the_masked_value() {
        let read = |_, _| Some(0x1_0000_0005);
        let w = |comparison, reference| ValueWait { address: 8, bytes: 4, comparison, reference, mask: 0xffff_ffff };
        let expect = [(NEVER, false), (LESS, false), (EQUAL, true), (LESS_EQUAL, true), (GREATER, false), (NOT_EQUAL, false), (GREATER_EQUAL, true), (ALWAYS, true)];
        for (cmp, holds) in expect {
            assert_eq!(w(cmp, 5).holds(&read), Some(holds), "comparison {cmp}");
        }
        assert_eq!(w(LESS, 6).holds(&read), Some(true));
        assert_eq!(w(GREATER, 4).holds(&read), Some(true));
        assert_eq!(w(0, 5).holds(&read), None, "unknown comparison");
        assert_eq!(w(9, 5).holds(&read), None);
        assert_eq!(w(EQUAL, 5).holds(&|_, _| None), None, "unreadable memory");
    }

    /// The compute queue's list waits for a value the graphics queue's list
    /// writes at its end: it runs after it.
    #[test]
    fn a_list_waits_for_another_queues_write() {
        let fake = RefCell::new(Fake::default());
        fake.borrow_mut().writes.insert("draw", vec![(0x100, 7)]);
        let mut held = Held::default();
        held.push(2, Item::Segment("depth", vec![ge(0x100, 7)]));
        let p = pump(&mut held, &fake);
        assert_eq!((p.pieces, p.segments), (0, 0), "held: the value is not there");
        assert!(held.holds_work(2));
        held.push(1, Item::Segment("draw", vec![]));
        let p = pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q1: draw", "q2: depth"]);
        assert_eq!((p.pieces, p.segments, p.waited.len()), (2, 2, 1));
        assert_eq!(p.waited[0].0, 2);
        assert!(!held.holds_work(2));
    }

    #[test]
    fn a_wait_that_holds_defers_nothing() {
        let fake = RefCell::new(Fake::default());
        fake.borrow_mut().memory.insert(0x100, 9);
        let mut held = Held::default();
        held.push(2, Item::Segment("a", vec![ge(0x100, 7)]));
        held.push(2, Item::Segment("b", vec![]));
        let p = pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q2: a b"], "one piece");
        assert_eq!((p.pieces, p.waited.len()), (1, 0));
    }

    /// A list's second segment waits for a value the other queue writes
    /// after reading what the first segment wrote: the cut is what lets
    /// both make progress.
    #[test]
    fn chained_waits_across_two_queues() {
        let fake = RefCell::new(Fake::default());
        {
            let mut f = fake.borrow_mut();
            f.writes.insert("c1", vec![(0x10, 1)]);
            f.writes.insert("g1", vec![(0x20, 1)]);
            f.writes.insert("c2", vec![(0x30, 1)]);
        }
        let mut held = Held::default();
        // Compute: [wait 0x08][c1, writes 0x10][wait 0x20][c2, writes 0x30].
        held.push(2, Item::Segment("c1", vec![ge(0x08, 1)]));
        held.push(2, Item::Segment("c2", vec![ge(0x20, 1)]));
        // Graphics: [g0, writes 0x08], [wait 0x10][g1, writes 0x20], [wait 0x30][g2].
        held.push(1, Item::Segment("g0", vec![]));
        fake.borrow_mut().writes.insert("g0", vec![(0x08, 1)]);
        held.push(1, Item::Segment("g1", vec![ge(0x10, 1)]));
        held.push(1, Item::Segment("g2", vec![ge(0x30, 1)]));
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q1: g0", "q2: c1", "q1: g1", "q2: c2", "q1: g2"]);
        assert!(held.blocked(&|_, _| Some(0)).is_empty());
    }

    /// Order within a queue never changes: a later segment whose waits hold
    /// does not pass a held one.
    #[test]
    fn a_queue_keeps_its_order() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::Segment("first", vec![ge(0x100, 1)]));
        held.push(2, Item::Segment("second", vec![]));
        pump(&mut held, &fake);
        assert!(fake.borrow().log.is_empty());
        fake.borrow_mut().memory.insert(0x100, 1);
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q2: first second"]);
    }

    /// A Signal behind held work is delivered after it: the title's fence
    /// (and its memory) does not move before the work ran.
    #[test]
    fn a_signal_waits_for_the_work_before_it() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::Segment("work", vec![ge(0x100, 1)]));
        held.push(2, Item::Signal(5, 3));
        held.push(2, Item::Segment("after", vec![]));
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().fences.get(&5), None, "not signalled yet");
        fake.borrow_mut().memory.insert(0x100, 1);
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q2: work", "signal 5 3", "q2: after"]);
    }

    #[test]
    fn a_fence_wait_holds_its_queue() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::FenceWait(5, 2));
        held.push(2, Item::Segment("after", vec![]));
        pump(&mut held, &fake);
        assert!(fake.borrow().log.is_empty());
        held.push(1, Item::Signal(5, 2));
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["signal 5 2", "q2: after"]);
    }

    /// The frame ends with work held: it stays held (and is reported), and
    /// the next frame's write releases it.
    #[test]
    fn work_held_at_the_frames_end_stays_held() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::Segment("late", vec![ge(0x100, 2)]));
        held.push(2, Item::Signal(5, 1));
        pump(&mut held, &fake);
        let read = |a: u64, _| Some(*fake.borrow().memory.get(&a).unwrap_or(&0));
        assert!(!held.is_empty());
        let blocked = held.blocked(&read);
        assert_eq!(blocked.len(), 1);
        assert_eq!((blocked[0].0, blocked[0].1, blocked[0].3.clone()), (2, 2, vec![ge(0x100, 2)]));
        fake.borrow_mut().memory.insert(0x100, 2);
        let p = pump(&mut held, &fake);
        assert_eq!(p.waited.len(), 1);
        assert!(held.is_empty());
        assert_eq!(fake.borrow().log, ["q2: late", "signal 5 1"]);
    }

    #[test]
    fn unknown_comparisons_are_counted_and_do_not_hold_work() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        let odd = ValueWait { address: 0x100, bytes: 4, comparison: 12, reference: 1, mask: 0xffff_ffff };
        held.push(2, Item::Segment("odd", vec![odd]));
        let p = pump(&mut held, &fake);
        assert_eq!((p.segments, p.unknown), (1, 1));
    }

    #[test]
    fn the_valve_releases_only_old_heads() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::Segment("stuck", vec![ge(0x100, 1)]));
        assert!(held.force_older_than(Duration::from_secs(60)).is_empty());
        let forced = held.force_older_than(Duration::ZERO);
        assert_eq!(forced, [(2, vec![ge(0x100, 1)])]);
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q2: stuck"]);
    }

    #[test]
    fn take_ready_marks_queue_running_until_finished() {
        let mut held = Held::default();
        held.push(1, Item::Segment("a", vec![]));
        held.push(1, Item::Signal(9, 1));
        held.push(1, Item::Segment("b", vec![]));
        let read = |_: u64, _| Some(0u64);
        let fence = |_: usize| 0u64;
        let no_write = |_: &&str, _: &ValueWait| false;
        let no_join = |_: &[(&str, bool)], _: &&str| false;

        match held.take_ready_with(&read, &fence, &no_write, &no_join, false) {
            Some(Ready::Segments {
                queues, segments, ..
            }) => {
                assert_eq!(queues, [1]);
                assert_eq!(segments, [("a", false)]);
            }
            _ => panic!("expected segments"),
        }
        // Queue 1 is marked running: holds_work stays true and take_ready_with
        // must not pop its queued Signal before finish_running(&[1]).
        assert!(held.holds_work(1));
        assert!(!held.is_empty());
        assert!(held
            .take_ready_with(&read, &fence, &no_write, &no_join, false)
            .is_none());

        held.finish_running(&[1]);
        match held.take_ready_with(&read, &fence, &no_write, &no_join, false) {
            Some(Ready::Signal { fence: f, value }) => assert_eq!((f, value), (9, 1)),
            _ => panic!("expected signal"),
        }
        match held.take_ready_with(&read, &fence, &no_write, &no_join, false) {
            Some(Ready::Segments {
                queues, segments, ..
            }) => {
                assert_eq!(queues, [1]);
                assert_eq!(segments, [("b", false)]);
            }
            _ => panic!("expected trailing segment"),
        }
        held.finish_running(&[1]);
        assert!(!held.holds_work(1));
        assert!(held.is_empty());
    }

    #[test]
    fn take_ready_coalesced_merges_cross_queue_bop_and_deferred_signals() {
        let mut held = Held::default();
        // Queue 2 waits on BoP 0x100 >= 1 written by Queue 1's segment "g1".
        held.push(2, Item::Segment("c1", vec![ge(0x100, 1)]));
        // Queue 3 waits on fence 9 >= 1 signalled after "g1".
        held.push(3, Item::FenceWait(9, 1));
        held.push(3, Item::Segment("c2", vec![]));
        // Queue 1 produces "g1" (writes 0x100 = 1), signals fence 9, then "g2".
        held.push(1, Item::Segment("g1", vec![]));
        held.push(1, Item::Signal(9, 1));
        held.push(1, Item::Segment("g2", vec![]));

        let read = |_: u64, _| Some(0u64);
        let fence = |_: usize| 0u64;
        let will_write = |seg: &&str, wait: &ValueWait| *seg == "g1" && wait.address == 0x100;
        let can_join = |_: &[(&str, bool)], next: &&str| *next != "c2";

        // First coalesced batch picks up Queue 1 ("g1", deferred Signal(9,1), "g2")
        // AND Queue 2 ("c1" via BoP with barrier_before=true), then splits before
        // Queue 3 ("c2") because `can_join` rejects it at the join boundary.
        match held.take_ready_with(&read, &fence, &will_write, &can_join, true) {
            Some(Ready::Segments {
                queues,
                segments,
                signals,
                coalesced_bop_waits,
                hazard_splits,
                ..
            }) => {
                assert_eq!(queues, [1, 2]);
                assert_eq!(segments, [("g1", false), ("g2", false), ("c1", true)]);
                assert_eq!(signals, [(9, 1)]);
                assert_eq!(coalesced_bop_waits, 1);
                assert_eq!(hazard_splits, 1);
            }
            _ => panic!("expected coalesced batch"),
        }
        assert!(held.holds_work(1));
        assert!(held.holds_work(2));
        held.finish_running(&[1, 2]);

        // Queue 3's FenceWait(9, 1) was restored when "c2" split out, so "c2"
        // stays blocked until fence 9 actually reaches 1.
        assert!(held
            .take_ready_with(&read, &fence, &will_write, &can_join, true)
            .is_none());

        // Once the first batch's signal (9, 1) has landed, Queue 3's "c2" runs.
        let fence_after = |f: usize| if f == 9 { 1 } else { 0 };
        match held.take_ready_with(&read, &fence_after, &will_write, &can_join, true) {
            Some(Ready::Segments {
                queues, segments, ..
            }) => {
                assert_eq!(queues, [3]);
                assert_eq!(segments, [("c2", false)]);
            }
            _ => panic!("expected c2 after fence"),
        }
        held.finish_running(&[3]);
        assert!(held.is_empty());
    }
}
