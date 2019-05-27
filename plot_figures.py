# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:04:06 2019

@author: xin-yi.song
"""

###################################折线图#######################################
#%matplotlib qt5###，在命令行中输入，用于在窗口显示图片

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,50)
y1 = 2*x+1
y2 = x**2

plt.figure()

l1, = plt.plot(x,y2,label='up')
l2, = plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--',label='down')

plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('I am x')
plt.ylabel('I am y')
new_ticks = np.linspace(-1,2,5)
#print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3,],
            ['$really\ bad$',r'$bad\ \alpha$',r'$normal$',r'$good$',r'$really\ good$'])
#显示中文，在后面加上fontproperties='FangSong'

#gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))#outward,axes百分比定位
ax.spines['left'].set_position(('data',0))

plt.legend(handles=[l1,l2,],labels=['aaa','bbb'],loc='best') #loc='center left','center right', 'lower left', 'lower right'
plt.show()

###################################散点图#######################################
import matplotlib.pyplot as plt
import numpy as np

n=1024
X=np.random.normal(0,1,n)
Y=np.random.normal(0,1,n)
T=np.arctan2(Y,X)#for color value

#plt.scatter(X,Y,s=75,c=T,alpha=0.5)
plt.scatter(np.arange(5),np.arange(5))#一条线上的散点图

#plt.xlim((-1.5,1.5))
#plt.ylim((-1.5,1.5))
plt.xticks(())#隐藏x标签
plt.yticks(())
plt.show()

##################################柱状图########################################
import matplotlib.pyplot as plt
import numpy as np

n=12
X=np.arange(n)
Y1=(1-X/float(n))*np.random.uniform(0.5,1.0,n)#均匀分布
Y2=(1-X/float(n))*np.random.uniform(0.5,1.0,n)

plt.bar(X,+Y1,facecolor='#9999ff',edgecolor='white')
plt.bar(X,-Y2,facecolor='#ff9999',edgecolor='white')

for x,y in zip(X,Y1):#zip打包成元组,zip在3.0版本返回的是生成器不能迭代，应该用z1=list(zip(x,y1))
    #ha:horizontal alignment
    plt.text(x+0.04,y+0.05,'%.2f'%y,ha='center',va='bottom')

for x,y in zip(X,Y2):#zip打包成元组,zip在3.0版本返回的是生成器不能迭代，应该用z1=list(zip(x,y1))
    #ha:horizontal alignment
    plt.text(x+0.04,-y-0.05,'-%.2f'%y,ha='center',va='top')

plt.xlim(-.5,n)
plt.xticks(())
plt.ylim(-1.25,1.25)
plt.yticks(())

plt.show()

##################################等高线图#####################################
import matplotlib.pyplot as plt
import numpy as np

def f(x,y):
    #the height function
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n=256
x=np.linspace(-3,3,n)
y=np.linspace(-3,3,n)
X,Y =np.meshgrid(x,y)#绑定网格

#use plt.contourf to filling contours
#X,Y and value for (X,Y) point
plt.contourf(X,Y,f(X,Y),8,alpha=0.75,cmap=plt.cm.hot)#alpha是透明度,hot可以改成cool,8代表分成10部分

#use plt.contour to add contour lines
C=plt.contour(X,Y,f(X,Y),8,colors='black',linewidths=.5)
#adding label
plt.clabel(C,inline=True,fontsize=10)

plt.xticks(())
plt.yticks(())
plt.show()

##################################Image图片###################################
import matplotlib.pyplot as plt
import numpy as np

#image data
a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)

"""
for the value of "interpolation", check this:
http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
for the value of "origin"= ['upper', 'lower'], check this:
http://matplotlib.org/examples/pylab_examples/image_origin.html
"""
    
plt.imshow(a,interpolation='nearest',cmap='bone',origin='lower')#lower可以改成upper
plt.colorbar(shrink=0.9)

plt.xticks()
plt.yticks()
plt.show()

##################################3D数据###################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()#窗口
ax = Axes3D(fig)#加坐标轴
#X,Y value
X= np.arange(-4,4,0.25)
Y= np.arange(-4,4,0.25)
X,Y = np.meshgrid(X,Y)
R = np.sqrt(X**2+Y**2)
#height value
Z = np.sin(R)

ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
#rstride行跨，cstride列跨
ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')#offset代表压倒坐标轴上哪个坐标
#ax.contourf(X,Y,Z,zdir='x',offset=-4,cmap='rainbow')
ax.set_zlim(-2,2)
plt.show()

##################################Subplot多合一显示################################
import matplotlib.pyplot as plt

plt.figure()

plt.subplot(2,1,1)
plt.plot([0,1],[0,1])

plt.subplot(2,3,4)
#plt.subplot(2,2,2)
plt.plot([0,1],[0,2])

plt.subplot(2,3,5)
#plt.subplot(2,2,3)
plt.plot([0,1],[0,3])

plt.subplot(2,3,6)
#plt.subplot(2,2,4)
plt.plot([0,1],[0,4])

plt.show()

##################################Subplot分格显示################################
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#method 1；subplot2grid
##########################
plt.figure()
ax1 = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=1)
ax1.plot([1,2],[1,2])
ax1.set_title('ax1_title')
ax2 = plt.subplot2grid((3,3),(1,0),colspan=2,)
ax3 = plt.subplot2grid((3,3),(1,2),roewpan=2)
ax4 = plt.subplot2grid((3,3),(2,0))
ax5 = plt.subplot2grid((3,3),(2,1))


#method 2:gridspec
##########################
plt.figure()
gs = gridspec.GridSpec(3,3)
ax1 = plt.subplot(gs[0,:])
ax2 = plt.subplot(gs[1,:2])
ax3 = plt.subplot(gs[-1,0])
ax5 = plt.subplot(gs[-1,-2])

#method 3:easy to define structure
###########################################
f,((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2,sharex=True,sharey=True)#共享x轴，y轴，改参数用f
ax11.scatter([1,2],[1,2])

plt.tight_layout()
plt.show()

######################################图中图######################################
import matplotlib.pyplot as plt

fig = plt.figure()
x = [1,2,3,4,5,6,7]
y = [1,3,4,2,5,8,6]

left,bottom,width,height = 0.1,0.1,0.8,0.8
ax1 = fig.add_axes([left,bottom,width,height])
ax1.plot(x,y,'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title')

left,bottom,width,height = 0.2,0.6,0.25,0.25
ax2 = fig.add_axes([left,bottom,width,height])
ax2.plot(y,x,'b')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('title inside 1')

plt.axes([.6,0.2,0.25,0.25])
plt.plot(y[::-1],x,'g')#y值逆序

plt.show()

######################################次坐标轴######################################
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,10,0.1)
y1 = 0.05*x**2
y2 = -1*y1

fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x,y1,'g-')
ax2.plot(x,y2,'b--')

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1',color='g')
ax2.set_ylabel('Y2',color='b')

plt.show()

########################################标注######################################
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,50)
y = 2*x+1

plt.figure(num=1,figsize=(8,5),)
plt.plot(x,y,linewidth=10,zorder=1)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

for label in ax.get_xticklabels()+ax.get_yticklabels():
  label.set_zorder(2)
  label.set_fontsize(12)
  label.set_bbox(dict(facecolor='white',edgecolor='None',alpha=0.7))#alpha代表透明度

x0 = 1
y0 = 2*x0+1
plt.scatter(x0,y0,s=50,color='b')
plt.plot([x0,x0],[y0,0],'k--',lw=2.5)

#method 1
################################
plt.annotate(r'$2x+1=%s$'%y0,xy=(x0,y0),xycoords='data',xytext=(+30,-30),textcoords='offset points',
             fontsize=16, arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))

#method 2
################################
plt.text(-3.7,3,r'$This\ is\ the\ some\ text.\ \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size':16, 'color':'r'})

plt.show()

########################################动画######################################
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig,ax = plt.subplots()

x = np.arange(0,2*np.pi,0.01)
line, = ax.plot(x,np.sin(x))#打逗号是为了选列表的第一个

def animate(i):
    line.set_ydata(np.sin(x+i/10))
    return line,

def init():
    line.set_ydata(np.sin(x))
    return line,

ani = animation.FuncAnimation(fig=fig,func=animate,frames=100,init_func=init,interval=20,blit=True)
plt.show()





