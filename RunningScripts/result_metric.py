def result_metric(sweeps, group_name, drop_list=[0], drop_HPT=False, metric=BASE_METRIC, epoch="best"):
    df = api.get_sweeps_results(sweeps, metric=metric)

    config_cols = [c for c in df.columns if
                   "config_" in c and c != "config_wandb_run_id" and c != "config_online_simulation_size"]
    HPT_cols = [col for col in config_cols if df[col].nunique() > 1]
    print(HPT_cols)
    if drop_HPT:
        df = df.drop([c for c in HPT_cols if not c in ["config_LLM_SIM_SIZE", "config_seed"]], axis=1)
        HPT_cols = ["config_LLM_SIM_SIZE", "config_seed"]

    # dropping all non-numeric columns that are not in the HPT_cols list
    drop_names = [c for c in df.columns if
                  (c not in HPT_cols) and (c not in [metric, "epoch"]) and (df[c].dtype not in [np.number])]
    df = df.drop(drop_names, axis=1)

    if len(HPT_cols) == 1:  # has only one column of config_seed
        # add a dummy column to group by
        df["dummy"] = 'dummy'
        HPT_cols.append("dummy")

    grouped = df.groupby([c for c in HPT_cols if c != "config_seed"], dropna=False)
    mean_df = grouped.mean()
    std_df = grouped.std()
    if epoch == "best":
        best_col = mean_df[
            [c for c in mean_df.columns if (metric in c and metric[-4:] == c.split("_epoch")[0][-4:])]].idxmax(axis=1)
    else:
        best_col = mean_df[[c for c in mean_df.columns if f"{metric}_epoch{epoch}" in c]].idxmax(axis=1)

    result = grouped.apply(lambda x: x[best_col.loc[x.name]].values)
    means = grouped.apply(lambda x: x[best_col.loc[x.name]].mean())
    stds = grouped.apply(lambda x: x[best_col.loc[x.name]].std())

    df_cols = {'mean': means, 'std': stds, 'values': result.values}
    if epoch == "best": df_cols['epoch'] = best_col.apply(
        lambda x: int(x.split("epoch")[1]) if "epoch" in x else "last")

    df_cols['CI'] = result.apply(lambda x: bootstrap_ci(x))

    summary_df = pd.DataFrame(df_cols, index=best_col.index)
    for d in drop_list:
        if d in summary_df.index:
            summary_df = summary_df.drop(d)
    if len(summary_df.index.names) == 1:
        return summary_df.rename_axis(group_name)
    else:
        return summary_df


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))
    lower_bound = np.percentile(bootstrapped_means, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_means, (1 + ci) / 2 * 100)
    return lower_bound, upper_bound