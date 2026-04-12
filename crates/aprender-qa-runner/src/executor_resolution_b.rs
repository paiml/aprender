impl Executor {

    /// Symlink a single model file and its sibling configs
    fn setup_single_file_link(source_file: &Path, st_dir: &Path) -> Option<String> {
        let st_link = st_dir.join("model.safetensors");
        let _ = std::fs::remove_file(&st_link);
        #[cfg(unix)]
        let link_result = std::os::unix::fs::symlink(source_file, &st_link);
        #[cfg(not(unix))]
        let link_result = std::fs::copy(source_file, &st_link).map(|_| ());

        if let Err(e) = link_result {
            return Some(format!("Failed to symlink model file: {e}"));
        }

        let siblings = Self::find_sibling_model_files(source_file);
        for (src_path, canonical_name) in &siblings {
            let link_path = st_dir.join(canonical_name);
            let _ = std::fs::remove_file(&link_path);
            #[cfg(unix)]
            let link_res = std::os::unix::fs::symlink(src_path, &link_path);
            #[cfg(not(unix))]
            let link_res = std::fs::copy(src_path, &link_path).map(|_| ());

            if let Err(e) = link_res {
                eprintln!(
                    "[WARN] Failed to link sibling {canonical_name}: {e}"
                );
            }
        }
        None
    }

    /// Convert source model to each requested non-SafeTensors format
    #[allow(clippy::too_many_arguments)]
    fn convert_requested_formats(
        &mut self,
        workspace: &Path,
        st_dir: &Path,
        source_file: &Path,
        model_id: &ModelId,
        requested_formats: &[Format],
        is_sharded: bool,
    ) -> (usize, usize) {
        let mut passed = 0;
        let mut failed = 0;

        for format in requested_formats {
            if *format == Format::SafeTensors {
                continue;
            }

            let (subdir, ext, gate_id) = match format {
                Format::Apr => ("apr", "apr", "G0-FORMAT-APR-001"),
                Format::Gguf => ("gguf", "gguf", "G0-FORMAT-GGUF-001"),
                Format::SafeTensors => unreachable!(),
            };

            let format_dir = workspace.join(subdir);
            if let Err(e) = std::fs::create_dir_all(&format_dir) {
                let ev = Evidence::falsified(
                    gate_id,
                    Self::format_scenario(model_id, *format),
                    format!("Failed to create {subdir} directory: {e}"),
                    "N/A",
                    0,
                );
                self.collector.add(ev);
                failed += 1;
                continue;
            }

            let target = format_dir.join(format!("model.{ext}"));
            let start = Instant::now();
            let convert_source = if is_sharded {
                st_dir.join(source_file.file_name().unwrap_or_default())
            } else {
                source_file.to_path_buf()
            };
            let output = self.command_runner.convert_model(&convert_source, &target);
            let duration = start.elapsed().as_millis() as u64;

            if output.success {
                let ev = Evidence::corroborated(
                    gate_id,
                    Self::format_scenario(model_id, *format),
                    &format!("G0 PASS: converted to {subdir}\n{}", output.stdout),
                    duration,
                );
                self.collector.add(ev);
                passed += 1;
            } else {
                let ev = Evidence::falsified(
                    gate_id,
                    Self::format_scenario(model_id, *format),
                    format!("G0 FAIL: conversion to {subdir} failed: {}", output.stderr),
                    &output.stdout,
                    duration,
                );
                self.collector.add(ev);
                failed += 1;
            }
        }

        (passed, failed)
    }

    /// Pre-flight gateway stub.
    ///
    /// Actual gateway enforcement is split across:
    /// - G0 sub-gates: checked in `execute()` via `run_g0_*` (Jidoka early returns)
    /// - G1-G4: evaluated post-execution by MQS scorer in `apr-qa-report`
    ///
    /// This hook exists for future pre-flight validation (e.g., model file
    /// existence, config.json accessibility) before heavy computation begins.
    fn check_gateways(&self, _playbook: &Playbook) -> Result<()> {
        Ok(())
    }

    /// Get collected evidence
    #[must_use]
    pub fn evidence(&self) -> &EvidenceCollector {
        &self.collector
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &ExecutionConfig {
        &self.config
    }
}
