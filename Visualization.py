# T-SNE可视化，画出不同分类的分布
# 二分类(积极和消极.但是因为还有中性样本，所以要做三分类)
def draw(result, label, way):
    result = torch.stack(result)
    tsne = TSNE(n_components=2 , init='pca' , perplexity=100,  random_state=2022, learning_rate = 150)
    result = tsne.fit_transform(result.cpu().detach().numpy())
    label = np.array(label)
    result0 = []
    result1 = []
    resultf1 = []
    for i, num in enumerate(label):
        if num == 0:
            result0.append(i)
        elif num == 1:
            result1.append(i)
        else:
            resultf1.append(i)
    r0 = result0
    r1 = result1
    rf1 = resultf1
    x_min, x_max = np.min(result), np.max(result)
    data = (result - x_min) / (x_max - x_min)
    x0 = [data[x, 0] for x in r0]
    y0 = [data[x, 1] for x in r0]
    x1 = [data[x, 0] for x in r1]
    y1 = [data[x, 1] for x in r1]
    xf1 = [data[x, 0] for x in rf1]
    yf1 = [data[x, 1] for x in rf1]

    plt.scatter(x1, y1, c='blue', label='g2', s = 10 , marker='s')
    plt.scatter(xf1, yf1, c='yellow', label='g3', s = 10 , marker='v')  
    plt.scatter(x0, y0, c='red', label='g1', s = 10 , marker='^')
    plt.rcParams.update({'font.size':11.5})
    plt.legend(['Positive', 'Negative', 'Neutral'], loc='lower left')
    
    from matplotlib.pyplot import MultipleLocator
    y_major_locator=MultipleLocator(0.2)
    ax=plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)      
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xticks(fontsize=16.5)
    plt.yticks(fontsize=16.5)
    plt.savefig(way)
    return fig 


# 七分类，情感强度从-3到3分为7类
            
def draw(result, label, way):
    result = torch.stack(result)
    tsne = TSNE(n_components=2 , init='pca' , perplexity=100,  random_state=2022, learning_rate = 150)
    result = tsne.fit_transform(result.cpu().detach().numpy())
    label = np.array(label)
    result0 = []
    result1 = []
    resultf1 = []
    resultf2 = []
    resultf3 = []
    result2 = []
    result3 = []
    for i, num in enumerate(label):
        if np.round(num) == 0:
            result0.append(i)
        elif np.round(num)  == 1:
            result1.append(i)
        elif np.round(num) == -1:
            resultf1.append(i)
        elif np.round(num) == -2:
            resultf2.append(i)
        elif np.round(num) == -3:
            resultf3.append(i)
        elif np.round(num)  == 2:
            result2.append(i)
        elif np.round(num)  == 3:
            result3.append(i)
            
            
    r0 = result0
    r1 = result1
    r2 = result2
    r3 = result3
    rf1 = resultf1
    rf2 = resultf2
    rf3 = resultf3
    x_min, x_max = np.min(result), np.max(result)
    data = (result - x_min) / (x_max - x_min)

    
    x0 = [data[x, 0] for x in r0]
    y0 = [data[x, 1] for x in r0]


    x1 = [data[x, 0] for x in r1]
    y1 = [data[x, 1] for x in r1]
    
    x2 = [data[x, 0] for x in r2]
    y2 = [data[x, 1] for x in r2]
    
    x3 = [data[x, 0] for x in r3]
    y3 = [data[x, 1] for x in r3]


    xf1 = [data[x, 0] for x in rf1]
    yf1 = [data[x, 1] for x in rf1]
    
    xf2 = [data[x, 0] for x in rf2]
    yf2 = [data[x, 1] for x in rf2]
    
    xf3 = [data[x, 0] for x in rf3]
    yf3 = [data[x, 1] for x in rf3]


    
    plt.scatter(x0, y0, c='red', label='g1', s = 10 , marker='^')
    plt.scatter(x1, y1, c='blue', label='g2', s = 10 , marker='+')
    plt.scatter(x2, y2, c='black', label='g4', s = 10 , marker='s')
    plt.scatter(x3, y3, c='green', label='g5', s = 10 , marker='D')
    plt.scatter(xf1, yf1, c='yellow', label='g3', s = 10 , marker='x')  
    plt.scatter(xf2, yf2, c='purple', label='g3', s = 10 , marker='>')  
    plt.scatter(xf3, yf3, c='gray', label='g3', s = 10 , marker='v')  
    
    plt.rcParams.update({'font.size':10})
    plt.legend(['0', '1', '2', '3', '-1', '-2', '-3'], loc='lower left')
    
    from matplotlib.pyplot import MultipleLocator
    y_major_locator=MultipleLocator(0.2)
    ax=plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)      
    plt.ylim(0, 1.0)
    plt.xlim(0, 1.0)
    plt.xticks(fontsize=16.5)
    plt.yticks(fontsize=16.5)
    
    plt.savefig(way)
    return fig 