use super::monitor_aspect::MonitorAspect;
use crate::cuda_funcs::CUDAEvent;
use crate::monitor::LaunchCUDAKernel;
use crate::monitor::error::MonitorError;
use crate::monitor::kernel_exec_time_aspect::QueryResult::Completed;
use anyhow::anyhow;
use object_pool::Pool;
use once_cell::sync::Lazy;
use serde::Serialize;
use std::cell::RefCell;
use std::sync::{Arc, Condvar};
use std::time::Duration;
use threadpool::ThreadPool;

struct Notification {
    pair: (std::sync::Mutex<bool>, Condvar),
}

impl Notification {
    fn new() -> Self {
        Notification {
            pair: (std::sync::Mutex::new(false), Condvar::new()),
        }
    }

    fn wait_for(&self, duration: Duration) -> bool {
        let expired = std::time::Instant::now() + duration;
        let (lock, cvar) = &self.pair;
        let mut started = lock.lock().unwrap();
        while !*started {
            let wait_duration = expired.saturating_duration_since(std::time::Instant::now());
            let result = cvar.wait_timeout(started, wait_duration).unwrap();
            started = result.0;
            if result.1.timed_out() {
                return false;
            }
        }
        true
    }

    fn notify(&self) {
        let (lock, cvar) = &self.pair;
        let mut started = lock.lock().unwrap();
        *started = true;
        cvar.notify_all();
    }
}

static EVENT_POOL: Lazy<Pool<CUDAEvent>> = Lazy::new(|| {
    Pool::new(8192, || {
        CUDAEvent::new().expect("Failed to create CUDAEvent")
    })
});

thread_local! {
    static LABEL: RefCell<String> = RefCell::new(String::new());
    static START_EVENT: RefCell<Option<CUDAEvent>> = RefCell::new(None);
    static USER_LABEL: RefCell<String> = RefCell::new(String::new());
}

pub struct KernelExecTimeAspect;

struct EventLogger {
    thread: ThreadPool,
    cancellation_token: Arc<Notification>,
}

enum QueryResult {
    Completed,
    Exited,
    Error,
}

fn query_event_with_notification(event: &CUDAEvent, token: &Notification) -> QueryResult {
    loop {
        match event.query() {
            Ok(true) => return QueryResult::Completed,
            Ok(false) => {
                if token.wait_for(Duration::from_millis(100)) {
                    return QueryResult::Exited;
                }
            }
            Err(err) => {
                log::error!("failed to query CUDA event: {}", err);
                return QueryResult::Error;
            }
        }
    }
}

#[derive(Serialize, Debug)]
#[serde(tag = "type", content = "data")]
enum LogMessage<'a> {
    Start {
        kern_label: &'a str,
        user_label: &'a str,
    },
    Complete {
        kern_label: &'a str,
        user_label: &'a str,
        duration_ms: f32,
    },
}

impl EventLogger {
    fn new() -> Self {
        let thread = ThreadPool::new(1);
        let cancellation_token = Arc::new(Notification::new());
        Self {
            thread,
            cancellation_token,
        }
    }

    fn add_event(&self, start: CUDAEvent, end: CUDAEvent, kern_label: String, user_label: String) {
        let cancellation_token = self.cancellation_token.clone();
        self.thread.execute(move || {
            match query_event_with_notification(&start, &cancellation_token) {
                Completed => {}
                _ => return,
            }

            log::info!(
                "{}",
                serde_json::to_string(&LogMessage::Start {
                    kern_label: kern_label.as_str(),
                    user_label: user_label.as_str(),
                })
                .expect("Failed to serialize CUDA event")
            );

            match query_event_with_notification(&end, &cancellation_token) {
                Completed => {}
                _ => return,
            }
            match end.since(&start) {
                Ok(duration) => {
                    log::info!(
                        "{}",
                        serde_json::to_string(&LogMessage::Complete {
                            kern_label: kern_label.as_str(),
                            user_label: user_label.as_str(),
                            duration_ms: duration,
                        })
                        .expect("Failed to serialize CUDA event")
                    );
                }
                Err(err) => {
                    log::error!("failed to compute elapsed time: {}", err);
                }
            }

            // return events to the pool
            EVENT_POOL.attach(start);
            EVENT_POOL.attach(end);
        });
    }
}

impl Drop for EventLogger {
    fn drop(&mut self) {
        self.cancellation_token.notify();
        self.thread.join();
    }
}

static EVENT_LOGGER: Lazy<EventLogger> = Lazy::new(|| EventLogger::new());

impl MonitorAspect for KernelExecTimeAspect {
    fn before_call(&self, launch: &LaunchCUDAKernel) -> Result<(), MonitorError> {
        START_EVENT.with(|se| -> Result<(), MonitorError> {
            let mut mut_se = se.borrow_mut();
            if mut_se.is_some() {
                return Err(MonitorError::Internal(anyhow!(
                    "START_EVENT is already set"
                )));
            }

            let (_, event) = EVENT_POOL
                .pull(|| CUDAEvent::new().expect("Failed to create CUDAEvent"))
                .detach();

            event
                .record(launch.stream())
                .map_err(MonitorError::CUDAError)?;

            mut_se.replace(event);

            LABEL.with(|label| label.replace(format!("{}", launch)));
            Ok(())
        })
    }

    fn after_call(&self, launch: &LaunchCUDAKernel) -> Result<(), MonitorError> {
        let mut ev = START_EVENT.replace(None);
        if ev.is_none() {
            return Err(MonitorError::Internal(anyhow!("START_EVENT is not set")));
        }

        let begin = ev.take().unwrap();
        let (_, end) = EVENT_POOL
            .pull(|| CUDAEvent::new().expect("Failed to create CUDAEvent"))
            .detach();
        end.record(launch.stream())
            .map_err(MonitorError::CUDAError)?;

        let label = LABEL.replace(String::new());

        EVENT_LOGGER.add_event(begin, end, label, USER_LABEL.with(|l| l.borrow().clone()));
        Ok(())
    }
}

pub fn set_kernel_exec_time_user_label(label: &str) {
    USER_LABEL.replace(label.to_string());
}
