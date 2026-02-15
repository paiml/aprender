impl CommandGenerator {
impl CommandGenerator {

    /// Generate synthetic commands
    pub fn generate(&self, count: usize) -> Vec<String> {
        let mut commands = Vec::with_capacity(count);
        let mut seen = HashSet::new();

        // Phase 1: Base commands from all templates (ensures diversity)
        for template in &self.templates {
            if seen.insert(template.base.to_string()) {
                commands.push(template.base.to_string());
            }
        }

        // Phase 2: Base + variant from all templates
        for template in &self.templates {
            for variant in &template.variants {
                let cmd = if variant.is_empty() {
                    template.base.to_string()
                } else {
                    format!("{} {}", template.base, variant)
                };
                if seen.insert(cmd.clone()) {
                    commands.push(cmd);
                }
            }
        }

        // Phase 3: Base + variant + flag from all templates
        for template in &self.templates {
            for variant in &template.variants {
                for flag in &template.flags {
                    let cmd = if variant.is_empty() {
                        format!("{} {}", template.base, flag)
                    } else {
                        format!("{} {} {}", template.base, variant, flag)
                    };
                    if seen.insert(cmd.clone()) {
                        commands.push(cmd);
                    }
                    if commands.len() >= count {
                        return commands;
                    }
                }
            }
        }

        // Phase 4: Base + variant + flag + arg (most expansive)
        for template in &self.templates {
            for variant in &template.variants {
                for flag in &template.flags {
                    for arg in &template.args {
                        let cmd = if variant.is_empty() {
                            format!("{} {} {}", template.base, flag, arg)
                        } else {
                            format!("{} {} {} {}", template.base, variant, flag, arg)
                        };
                        if seen.insert(cmd.clone()) {
                            commands.push(cmd);
                        }
                        if commands.len() >= count {
                            return commands;
                        }
                    }
                }
            }
        }

        commands.truncate(count);
        commands
    }
}
}
