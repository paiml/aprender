//! Purpose-built GEMV thread pool with barrier synchronization for quantized matmul dispatch.

// PMAT-310: Purpose-built GEMV thread pool with barrier synchronization.
// Replaces rayon for matmul dispatch. Zero work-stealing overhead.

use std::sync::{Arc, Barrier, Mutex};

type WorkFn = Box<dyn Fn(usize, usize) + Send + Sync>;

struct SharedWork {
    work: Mutex<Option<(WorkFn, usize)>>, // (function, out_dim)
}

pub(crate) struct GemvPool {
    n_threads: usize,
    shared: Arc<SharedWork>,
    start_barrier: Arc<Barrier>,
    end_barrier: Arc<Barrier>,
    _handles: Vec<std::thread::JoinHandle<()>>,
}

impl GemvPool {
    pub fn new(n: usize) -> Self {
        let shared = Arc::new(SharedWork {
            work: Mutex::new(None),
        });
        let start_barrier = Arc::new(Barrier::new(n + 1));
        let end_barrier = Arc::new(Barrier::new(n + 1));

        let mut handles = Vec::with_capacity(n);
        for tid in 0..n {
            let s = Arc::clone(&shared);
            let sb = Arc::clone(&start_barrier);
            let eb = Arc::clone(&end_barrier);
            let nt = n;
            handles.push(
                std::thread::Builder::new()
                    .name(format!("gemv-{tid}"))
                    .spawn(move || {
                        loop {
                            sb.wait();
                            let (f, out_dim) = {
                                let guard = s.work.lock().unwrap();
                                match guard.as_ref() {
                                    Some((f, od)) => {
                                        // Can't clone Box<dyn Fn>, but we can get a ref
                                        // The Mutex is only locked briefly to read the ptr
                                        let f_ptr = f.as_ref()
                                            as *const (dyn Fn(usize, usize) + Send + Sync);
                                        (f_ptr, *od)
                                    },
                                    None => {
                                        eb.wait();
                                        return; // shutdown
                                    },
                                }
                            };
                            let rows_per = (out_dim + nt - 1) / nt;
                            let start = tid * rows_per;
                            let end = (start + rows_per).min(out_dim);
                            if start < out_dim {
                                // SAFETY: f is valid for the duration of dispatch()
                                // (caller blocks on end_barrier before dropping the closure)
                                unsafe { (*f)(start, end) };
                            }
                            eb.wait();
                        }
                    })
                    .expect("spawn gemv thread"),
            );
        }

        Self {
            n_threads: n,
            shared,
            start_barrier,
            end_barrier,
            _handles: handles,
        }
    }

    #[inline]
    pub fn dispatch<F: Fn(usize, usize) + Send + Sync>(&self, out_dim: usize, f: &F) {
        // Box the closure reference (we know f lives until end_barrier.wait)
        let f_ref: &(dyn Fn(usize, usize) + Send + Sync) = f;
        // SAFETY: f lives until we call end_barrier.wait() below
        let f_static: &'static (dyn Fn(usize, usize) + Send + Sync) =
            unsafe { std::mem::transmute(f_ref) };
        let boxed: WorkFn = Box::new(move |a, b| (*f_static)(a, b));

        {
            let mut guard = self.shared.work.lock().unwrap();
            *guard = Some((boxed, out_dim));
        }

        self.start_barrier.wait();
        self.end_barrier.wait();

        // Clear work
        {
            let mut guard = self.shared.work.lock().unwrap();
            *guard = None;
        }
    }
}

impl Drop for GemvPool {
    fn drop(&mut self) {
        {
            let mut guard = self.shared.work.lock().unwrap();
            *guard = None;
        }
        self.start_barrier.wait();
        self.end_barrier.wait();
    }
}

static POOL: std::sync::OnceLock<GemvPool> = std::sync::OnceLock::new();

pub(crate) fn get_pool() -> &'static GemvPool {
    POOL.get_or_init(|| GemvPool::new(rayon::current_num_threads()))
}
