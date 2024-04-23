#Detecting outlier

#percentile based outlier detection
def percentile_outlier(data,thresh=95):
    diff=(100-thresh) / 2
    (min,max)=np.percentile(data,[diff,100-diff])
    return((data<min) | (data>max))



#median based outlier
def mad_based_outlier(points,thresh=3.5):
    if len(points.shape)==1:
        points = points.reshape(-1, 1)
    median_y=np.median(points)
    median_absolute_deviation_y=np.median(np.abs(y*median_y) for y in points)
    modified_z_scores=[0.6745*(y-median_y)/median_absolute_deviation_y for y in points]

    return np.abs(modified_z_scores)


def std_div(data,thresh=3):
    std=data.std()
    mean=data.mean()
    Outlier=[]
    for val in data:
        if val/std>thresh:
            Outlier.append(True)
        else:
            Outlier.append(False)
    return Outlier


def outliervote(data):
    x=percentile_outlier
    y=mad_based_outlier
    z=std_div
    temp=zip(data.index,x,y,z)
    final=[]
    for i in range(len(temp)):
        if temp[i].count(False)>=2:
            final.append(False)
        else:
            final.append(True)
        return final


def plotOutlier(x):
    fig, axes = plt.subplots(nrows=4)
    for ax, func in zip(axes, [percentile_outlier, mad_based_outlier, std_div, outliervote]):
        sns.distplot(x, ax=ax, rug=True, hist=False)
        outliers = x[func(x)]
        ax.plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)
    
    kwargs = dict(y=0.95, x=0.05, ha='left', va='top', size=20)
    axes[0].set_title('Percentile-based Outliers', **kwargs)
    axes[1].set_title('MAD-based Outliers', **kwargs)
    axes[2].set_title('STD-based Outliers', **kwargs)
    axes[3].set_title('Majority vote based Outliers', **kwargs)
    fig.suptitle('Comparing Outlier Tests with n={}'.format(len(x)), size=20)
    fig.set_size_inches(15, 10)
    plt.show()
