use tracel_xtask::prelude::*;

#[macros::extend_command_args(BuildCmdArgs, Target, None)]
pub struct CubeCLBuildCmdArgs {
    /// Build in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    mut args: CubeCLBuildCmdArgs,
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
    base_commands::build::handle_command(args.try_into().unwrap(), env, context)?;
    // Specific additional commands to test specific features
    helpers::custom_crates_build(
        vec!["t4a-cubecl-runtime"],
        vec!["--no-default-features"],
        None,
        None,
        "without default features",
    )?;

    Ok(())
}
