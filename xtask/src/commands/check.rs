use tracel_xtask::prelude::*;

#[macros::extend_command_args(CheckCmdArgs, Target, CheckSubCommand)]
pub struct CubeCLCheckCmdArgs {
    /// Build in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    mut args: CubeCLCheckCmdArgs,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
    if args.ci {
        // Exclude crates that are not supported on CI
        args.exclude.extend(vec![
            "t4a-cubecl-cuda".to_string(),
            "cubecl-hip".to_string(),
        ]);
    }
    base_commands::check::handle_command(args.try_into().unwrap(), env, context)?;
    // Specific additional commands to test specific features
    // cubecl-wgpu with exclusive-memory-only
    // cubecl-runtime without default features
    helpers::custom_crates_check(
        vec!["t4a-cubecl-wgpu"],
        vec!["--features", "exclusive-memory-only"],
        None,
        None,
        "std with exclusive_memory_only",
    )?;
    helpers::custom_crates_check(
        vec!["t4a-cubecl-runtime"],
        vec!["--no-default-features"],
        None,
        None,
        "without default features",
    )?;

    Ok(())
}
