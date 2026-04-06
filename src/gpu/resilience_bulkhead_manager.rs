
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bulkhead_manager_acquire_release() {
        let config = BulkheadConfig::new().with_pool("inference", 2);
        let manager = BulkheadManager::new(&config);

        let permit1 = manager.acquire(RequestType::Inference).expect("permit1");
        assert_eq!(manager.available(RequestType::Inference), 1);

        let permit2 = manager.acquire(RequestType::Inference).expect("permit2");
        assert_eq!(manager.available(RequestType::Inference), 0);

        manager.release(&permit1);
        assert_eq!(manager.available(RequestType::Inference), 1);

        manager.release(&permit2);
        assert_eq!(manager.available(RequestType::Inference), 2);
    }

    #[test]
    fn test_bulkhead_manager_acquire_exhausted() {
        let config = BulkheadConfig::new().with_pool("batch", 1);
        let manager = BulkheadManager::new(&config);

        let _permit = manager.acquire(RequestType::Batch).expect("_permit");
        let result = manager.acquire(RequestType::Batch);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Pool exhausted");
    }

    #[test]
    fn test_bulkhead_manager_try_acquire() {
        let config = BulkheadConfig::new().with_pool("embedding", 1);
        let manager = BulkheadManager::new(&config);

        let permit = manager.try_acquire(RequestType::Embedding).expect("permit");
        assert_eq!(manager.available(RequestType::Embedding), 0);

        let result = manager.try_acquire(RequestType::Embedding);
        assert!(result.is_err());

        manager.release(&permit);
        assert_eq!(manager.available(RequestType::Embedding), 1);
    }

    #[test]
    fn test_bulkhead_manager_stats() {
        let config = BulkheadConfig::new()
            .with_pool("inference", 10)
            .with_pool("embedding", 5)
            .with_pool("batch", 2);
        let manager = BulkheadManager::new(&config);

        let stats = manager.stats();
        assert_eq!(stats.pool_count, 3);
        assert_eq!(stats.total_capacity, 17);
    }

    #[test]
    fn test_bulkhead_manager_isolation() {
        let config = BulkheadConfig::new()
            .with_pool("inference", 2)
            .with_pool("embedding", 2);
        let manager = BulkheadManager::new(&config);

        // Exhaust inference pool
        let _p1 = manager.acquire(RequestType::Inference).expect("_p1");
        let _p2 = manager.acquire(RequestType::Inference).expect("_p2");

        // Embedding pool should still be available
        let result = manager.acquire(RequestType::Embedding);
        assert!(result.is_ok());
    }
include!("resilience_error_category.rs");
}
