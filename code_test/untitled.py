def equalObs(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))
def linreg_by_bin(df_2_vars, nbins = 10, ax = None, label = None, equal_size = True, **args):
    x = df_2_vars.iloc[:,0].values
    if equal_size:
        x_bins = equalObs(x, nbins) # bins of ~ equal size
    else:
        x_bins = np.linspace(np.min(x), np.max(x), nbins)
    results = []
    for i in range(len(x_bins)-1):
        df_bin = df_2_vars.loc[((x < x_bins[i+1])&
                                  (x > x_bins[i])),:]
        if len(df_bin)>50:
            x_middle = x_bins[i]+(x_bins[i+1] - x_bins[i])/2
            y_median = np.median(df_bin.iloc[:,1].values)
            if equal_size:
                x_middle = np.median(df_bin.iloc[:,0].values)
            y_10pct = np.percentile(df_bin.iloc[:,1].values, 25)
            y_90pct = np.percentile(df_bin.iloc[:,1].values, 75)
        else: x_middle, y_median, y_10pct, y_90pct= [np.nan, np.nan, np.nan, np.nan]
        results.append([x_middle, y_median, y_10pct, y_90pct])
    results = pd.DataFrame(data=results,columns = [‘x_middle’, ‘y_median’, ‘y_10pct’, ‘y_90pct’])
    if ax is None:
        f,ax = plt.subplots()
    ax.fill_between(x=results.x_middle, y1=results.y_10pct, y2=results.y_90pct, color= ‘lightgrey’, alpha = 0.1)
    sns.regplot(x=results.x_middle, y = results.y_median, marker = ‘o’, ax=ax, ci=0, label=label)
    reg_binned = LinearRegression().fit(results.dropna().x_middle.values.reshape(-1,1), results.dropna().y_median.values.reshape(-1,1))
    ax.text(0.2, 0.8, f”coeff = {np.round(reg_binned.coef_[0][0],3)}“, transform=ax.transAxes)
    ax.set_xlabel(df_2_vars.columns[0])
    ax.set_ylabel(df_2_vars.columns[1])
    if label is not None:
        plt.legend()
    return results