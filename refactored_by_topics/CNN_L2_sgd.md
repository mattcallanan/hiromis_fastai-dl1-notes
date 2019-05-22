# SGD [[1:15:06](https://youtu.be/Egp4Zajhzog?t=4506)]

So let's jump into a notebook and generate some dots, and see if we can get it to fit a line somehow. And the "somehow" is going to be using something called SGD. What is SGD? Well, there's two types of SGD. The first one is where I said in lesson 1 "hey you should all try building these models and try and come up with something cool" and you guys all experimented and found really good stuff. So that's where the S would be Student. That would be Student Gradient Descent. So that's version one of SGD.

Version two of SGD which is what I'm going to talk about today is where we are going to have a computer try lots of things and try and come up with a really good function and that would be called Stochastic Gradient Descent. The other one that you hear a lot on Twitter is Stochastic Grad student Descent.



#### Linear Regression problem [[1:16:08](https://youtu.be/Egp4Zajhzog?t=4568)]

We are going to jump into [lesson2-sgd.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-sgd.ipynb). We are going to go bottom-up rather than top-down. We are going to create the simplest possible model we can which is going to be a linear model. And the first thing we need is we need some data. So we are going to generate some data. The data we're going to generate looks like this:

 ![](../lesson2/n1.png)

So x-axis might represent temperature, y-axis might represent number of ice creams we sell, or something like that. But we're just going to create some synthetic data that we know is following a line. As we build this, we're actually going to lean a little bit about PyTorch as well.

```python
%matplotlib inline
from fastai import *
```

```python
n=100
```

```python
x = torch.ones(n,2)
x[:,0].uniform_(-1.,1)
x[:5]
```

```
tensor([[-0.1338,  1.0000],
        [-0.4062,  1.0000],
        [-0.3621,  1.0000],
        [ 0.4551,  1.0000],
        [-0.8161,  1.0000]])
```



Basically the way we're going to generate this data is by creating some coefficients.  <img src="http://latex.codecogs.com/gif.latex?a_1" title="a_1" /> will be 3 and <img src="http://latex.codecogs.com/gif.latex?a_2" title="a_2" /> will be 2. We are going to create a column of numbers for our <img src="http://latex.codecogs.com/gif.latex?x" title="x" />'s and a whole bunch of 1's.

```python
a = tensor(3.,2); a
```

```
tensor([3., 2.])
```



And then we're going to do this `x@a`. What is `x@a`? `x@a` in Python means a matrix product between `x` and `a`.  And it actually is even more general than that. It can be a vector vector product, a matrix vector product, a vector matrix product, or a matrix matrix product. Then actually in PyTorch, specifically, it can mean even more general things where we get into higher rank tensors which we will learn all about very soon. But this is basically the key thing that's going to go on in all of our deep learning. The vast majority of the time, our computers are going to be basically doing this﹣multiplying numbers together and adding them up which is the surprisingly useful thing to do.

```python
y = x@a + torch.rand(n)
```



[[1:17:57](https://youtu.be/Egp4Zajhzog?t=4677)]

So we basically are going to generate some data by creating a line and then we're going to add some random numbers to it. But let's go back and see how we created `x` and `a`. I mentioned that we've basically got these two coefficients 3 and 2. And you'll see that we've wrapped it in this function called `tensor`. You might have heard this word "tensor" before. It's one of these words that sounds scary and apparently if you're a physicist, it actually is scary. But in the world of deep learning, it's actually not scary at all. "tensor" means array, but specifically it's an array of a regular shape. So it's not an array where row 1 has two things, row 3 has three things, and row 4 has one thing, what you call a "jagged array". That's not a tensor.  A tensor is any array which has a rectangular or cube or whatever ﹣ a shape where every row is the same length and every column is the same length. The following are all tensors:

- 4 by 3 matrix
- A vector of length 4
- A 3D array of length 3 by 4 by 6

That's all tensor is. We have these all the time. For example, an image is a 3 dimensional tensor. It's got number of rows by number of columns by number of channels (normally red, green, blue). So for example, VGA picture could be 640 by 480 by 3 or actually we do things backwards so when people talk about images, they normally go width by height, but when we talk mathematically, we always go a number of rows by number of columns, so it would actually be 480 by 640 by 3 that will catch you out. We don't say dimensions, though, with tensors. We use one of two words, we either say rank or axis. Rank specifically means how many axes are there, how many dimensions are there. So an image is generally a rank 3 tensor. What we created here is a rank 1 tensor (also known as a vector). But in math, people come up with very different words for slightly different concepts. Why is a one dimensional array a vector and a two dimensional array is a matrix, and a three dimensional array doesn't have a name. It doesn't make any sense. With computers, we try to have some simple consistent naming conventions. They are all called tensors﹣rank 1 tensor, rank 2 tensor, rank 3 tensor. You can certainly have a rank 4 tensor. If you've got 64 images, then that would be a rank 4 tensor of 64 by 480 by 640 by 3. So tensors are very simple. They just mean arrays.



In PyTorch, you say `tensor` and you pass in some numbers, and you get back, which in this case just a list,  a vector. This then represents our coefficients: the slope and the intercept of our line.

![](../lesson2/28.png)

Because we are not actually going to have a special case of <img src="http://latex.codecogs.com/gif.latex?ax&space;&plus;&space;b" title="ax + b" />, instead, we are going to say there's always this second <img src="http://latex.codecogs.com/gif.latex?x" title="x" /> value which is always 1

<img src="http://latex.codecogs.com/gif.latex?y_i&space;=&space;a_1x_i_,_1&space;&plus;&space;a_2x_i_,_2" title="y_i = a_1x_i_,_1 + a_2x_i_,_2" />

You can see it here, always 1 which allows us just to do a simple matrix vector product:

![](../lesson2/29.png)

So that's <img src="http://latex.codecogs.com/gif.latex?a" title="a" />. Then we wanted to generate this <img src="http://latex.codecogs.com/gif.latex?x" title="x" /> array of data. We're going to put random numbers in the first column and a whole bunch of 1's in the second column. To do that, we say to PyTorch that we want to create a rank 2 tensor of `n` by 2. Since we passed in a total of 2 things, we get a rank 2 tensor. The number of rows will be `n` and the number of columns will be 2. In there, every single thing in it will be a 1﹣that's what `torch.ones` means.

[[1:22:45](https://youtu.be/Egp4Zajhzog?t=4965)]

Then this is really important. You can index into that just like you can index into a list in Python. But you can put a colon anywhere and a colon means every single value on that axis/dimension. This here `x[:,0]` means every single row of column 0. So `x[:,0].uniform_(-1.,1)` is every row of column 0, I want you to grab a uniform random numbers.

Here is another very important concept in PyTorch. Anytime you've got a function that ends with an underscore, it means don't return to me that uniform random number, but replace whatever this is being called on with the result of this function.  So this `x[:,0].uniform_(-1.,1)` takes column 0 and replaces it with a uniform random number between -1 and 1. So there's a lot to unpack there.

![](../lesson2/30.png)

But the good news is these two lines of code and `x@a` which we are coming to cover 95% of what you need to know about PyTorch.

1. How to create an array
2. How to change things in an array
3. How to do matrix operations on an array

So there's a lot to unpack, but these small number of concepts are incredibly powerful. So I can now print out the first five rows. `[:5]` is a standard Python slicing syntax to say the first 5 rows. So here are the first 5 rows, 2 columns looking like﹣my random numbers and my 1's.

Now I can do a matrix product of that `x` by my `a`, add in some random numbers to add a bit of noise.  

```python
y = x@a + torch.rand(n)
```

Then I can do a scatter plot. I'm not really interested in my scatter plot in this column of ones. They are just there to make my linear function more convenient. So I'm just going to plot my zero index column against my `y`'s.  

```python
plt.scatter(x[:,0], y);
```

 ![](../lesson2/n1.png)

`plt` is what we universally use to refer to the plotting library, matplotlib. That's what most people use for most of their plotting in scientific Python. It's certainly a library you'll want to get familiar with because being able to plot things is really important. There are lots of other plotting packages. Lots of the other packages are better at certain things than matplotlib, but matplotlib can do everything reasonably well. Sometimes it's a little awkward, but for me, I do pretty much everything in matplotlib because there is really nothing it can't do even though some libraries can do other things a little bit better or prettier. But it's really powerful so once you know matplotlib, you can do everything. So here, I'm asking matplotlib to give me a scatterplot with my `x`'s against my `y`'s. So this is my dummy data representing temperature and ice cream sales.



[[1:26:18]](https://youtu.be/Egp4Zajhzog?t=5178)

Now what we're going to do is, we are going to pretend we were given this data and we don't know that the values of our coefficients are 3 and 2. So we're going to pretend that we never knew that and we have to figure them out. How would we figure them out? How would we draw a line to fit this data and why would that even be interesting? Well, we're going to look at more about why it's interesting in just a moment. But the basic idea is:

>If we can find a way to find those two parameters to fit that line to those 100 points, we can also fit these arbitrary functions that convert from pixel values to probabilities.

It will turn out that this techniques that we're going to learn to find these two numbers works equally well for the 50 million numbers in ResNet34. So we're actually going to use an almost identical approach. This is the bit that I found in previous classes people have the most trouble digesting. I often find, even after week 4 or week 5, people will come up to me and say:

Student: I don't get it. How do we actually train these models?

Jeremy: It's SGD. It's that thing we saw in the notebook with the 2 numbers.

Student: yeah, but... but we are fitting a neural network.

Jeremy: I know and we can't print the 50 million numbers anymore, but it's literally identically doing the same thing.

The reason this is hard to digest is that the human brain has a lot of trouble conceptualizing of what an equation with 50 million numbers looks like and can do. So for now, you'll have to take my word for it. It can do things like recognize teddy bears. All these functions turn out to be very powerful. We're going to learn about how to make them extra powerful. But for now, this thing we're going to learn to fit these two numbers is the same thing that we've just been using to fit 50 million numbers.



### Loss function [[1:28:36](https://youtu.be/Egp4Zajhzog?t=5316)]

We want to find what PyTorch calls **parameters**, or in statistics, you'll often hear it called coefficient (i.e. these values of <img src="http://latex.codecogs.com/gif.latex?a_1" title="a_1" /> and <img src="http://latex.codecogs.com/gif.latex?a_2" title="a_2" />). We want to find these parameters such that the line that they create minimizes the error between that line and the points. In other words, if the <img src="http://latex.codecogs.com/gif.latex?a_1" title="a_1" /> and <img src="http://latex.codecogs.com/gif.latex?a_2" title="a_2" /> we came up with resulted in this line:

![](../lesson2/31.png)

Then we'd look and we'd see how far away is that line from each point. That's quite a long way. So maybe there was some other <img src="http://latex.codecogs.com/gif.latex?a_1" title="a_1" /> and <img src="http://latex.codecogs.com/gif.latex?a_2" title="a_2" /> which resulted in the gray line. And they would say how far away is each of those points. And then eventually we come up with the yellow line. In this case, each of those is actually very close.

![](../lesson2/32.png)

So you can see how in each case we can say how far away is the line at each spot away from its point, and then we can take the average of all those. That's called the **loss**. That is the value of our loss. So you need a mathematical function that can basically say how far away is this line from those points.

For this kind of problem which is called a regression problem (a problem where your dependent variable is continuous, so rather than being grizzlies or teddies, it's some number between -1 and 6), the most common loss function is called mean squared error which pretty much everybody calls MSE. You may also see RMSE which is root mean squared error. The mean squared error is a loss which is the difference between some predictions that you made which is like the value of the line and the actual number of ice cream sales. In the mathematics of this, people normally refer to the actual as <img src="http://latex.codecogs.com/gif.latex?y" title="y" /> and the prediction, they normally call it <img src="https://latex.codecogs.com/gif.latex?\hat{y}" title="\hat{y}" /> (y hat).  

When writing something like mean squared error equation, there is no point writing "ice cream" and "temperature" because we want it to apply to anything. So we tend to use these mathematical placeholders.

So the value of mean squared error is simply the difference between those two (`y_hat-y`) squared. Then we can take the mean because both `y_hat` and `y ` are rank 1 tensors, so we subtract one vector from another vector, it does something called "element-wise arithmetic" in other words, it subtracts each one from each other, so we end up with a vector of differences. Then if we take the square of that, it squares everything in that vector. So then we can take the mean of that to find the average square of the differences between the actuals and the predictions.

```python
def mse(y_hat, y): return ((y_hat-y)**2).mean()
```

If you're more comfortable with mathematical notation, what we just wrote was:

<img src="https://latex.codecogs.com/gif.latex?\frac{\sum&space;(\hat{y}-y)^2}{n}" title="\frac{\sum (\hat{y}-y)^2}{n}" />

One of the things I'll note here is, I don't think `((y_hat-y)**2).mean()` is more complicated or unwieldy than <img src="https://latex.codecogs.com/gif.latex?\frac{\sum&space;(\hat{y}-y)^2}{n}" title="\frac{\sum (\hat{y}-y)^2}{n}" /> but the benefit of the code is you can experiment with it. Once you've defined it, you can use it, you can send things into it, get stuff out of it, and see how it works. So for me, most of the time, I prefer to explain things with code rather than with math. Because they are the same, just different notations. But one of the notations is executable. It's something you can experiment with. And the other is abstract. That's why I'm generally going to show code.

So the good news is, if you're a coder with not much of a math background, actually you do have a math background. Because code is math. If you've got more of a math background and less of a code background, then actually a lot of the stuff that you learned from math is going to translate directly into code and now you can start to experiment with your math.



[[1:34:03](https://youtu.be/Egp4Zajhzog?t=5643)]

 `mse` is a loss function. This is something that tells us how good our line is. Now we have to come up with what is the line that fits through here. Remember, we are going to pretend we don't know. So what you actually have to do is you have to guess. You actually have to come up with a guess what are the values of <img src="http://latex.codecogs.com/gif.latex?a_1" title="a_1" /> and <img src="http://latex.codecogs.com/gif.latex?a_2" title="a_2" />. So let's say we guess that <img src="http://latex.codecogs.com/gif.latex?a_1" title="a_1" /> and <img src="http://latex.codecogs.com/gif.latex?a_2" title="a_2" /> are -1 and 1.

```python
a = tensor(-1.,1)
```

Here is how you create that tensor and I wanted to write it this way because you'll see this all the time. Written out fully, it would be `tensor(-1.0, 1.0)`. We can't write it without the point because `tensor(-1, 1)` is now an int, not a floating point. So that's going to spit the dummy (Australian for "behave in a bad-tempered or petulant way") if you try to do calculations with that in neural nets.

I'm far too lazy to type `.0` every time. Python knows perfectly well that if you added `.` next to any of these numbers, then the whole thing is now floats. So that's why you'll often see it written this way, particularly by lazy people like me.

So `a` is a tensor. You can see it's floating-point. You see, even PyTorch is lazy. They just put a dot. They don't bother with a zero.

![](../lesson2/33.png)

But if you want to actually see exactly what it is, you can write `.type()` and you can see it's a FloatTensor:

![](../lesson2/34.png)

So now we can calculate our predictions with this random guess. `x@a` a matrix product of `x` and `a`. And we can now calculate the mean squared error of our predictions and their actuals, and that's our loss. So for this regression, our loss is 0.9.

```python
y_hat = x@a
mse(y_hat, y)
```

```
tensor(8.8945)
```



So we can now plot a scatter plot of `x` against `y` and we can plot the scatter plot of `x` against `y_hat`.  And there they are.

```python
plt.scatter(x[:,0],y)
plt.scatter(x[:,0],y_hat);
```

![](../lesson2/n2.png)

 So that is not great﹣not surprising. It's just a guess. So SGD or gradient descent more generally and anybody who's done engineering or probably computer science at school would have done plenty of this like Newton's method, etc at university. If you didn't, don't worry. We're going to learn it now.

It's basically about taking this guess and trying to make it a little bit better. How do we make it a little better? Well, there are only two numbers and the two numbers are the two numbers are the intercept of the orange line and the gradient of the orange line. So what we are going to do with gradient descent is we're going to simply say:

- What if we changed those two numbers a little bit?
  - What if we made the intercept a little bit higher or a little bit lower?
  - What if we made the gradient a little bit more positive or a little bit more negative?

![](../lesson2/35.png)

There are 4 possibilities and then we can calculate the loss for each of those 4 possibilities and see what works. Did lifting it up or down make it better? Did tilting it more positive or more negative make it better? And then all we do is we say, okay, whichever one of those made it better, that's what we're going to do. That's it.

But here is the cool thing for those of you that remember calculus. You don't actually have to move it up and down, and round about. You can actually calculate the derivative. The derivative is the thing that tells you would moving it up or down make it better, or would rotating it this way or that way make it better. The good news is if you didn't do calculus or you don't remember calculus, I just told you everything you need to know about it. It tells you how changing one thing changes the function. That's what the derivative is, kind of, not quite strictly speaking, but close enough, also called the gradient. The gradient or the derivative tells you how changing <img src="http://latex.codecogs.com/gif.latex?a_1" title="a_1" /> up or down would change our MSE, how changing <img src="http://latex.codecogs.com/gif.latex?a_2" title="a_2" /> up or down would change our MSE, and this does it more quickly than actually moving it up and down.

In school, unfortunately, they forced us to sit there and calculate these derivatives by hand. We have computers. Computers can do that for us. We are not going to calculate them by hand.

```python
a = nn.Parameter(a); a
```

```
Parameter containing:
tensor([-1.,  1.], requires_grad=True)
```



[[1:39:12]](https://youtu.be/Egp4Zajhzog?t=5952)

Instead, we're doing to call `.grad`. On our computer, that will calculate the gradient for us.  

```python
def update():
    y_hat = x@a
    loss = mse(y, y_hat)
    if t % 10 == 0: print(loss)
    loss.backward()
    with torch.no_grad():
        a.sub_(lr * a.grad)
        a.grad.zero_()
```

So here is what we're going to do. We are going to create a loop. We're going to loop through 100 times, and we're going to call a function called `update`. That function is going to:

- Calculate `y_hat` (i.e. our prediction)

- Calculate loss (i.e. our mean squared error)

- From time to time, it will print that out so we can see how we're going

- Calculate the gradient. In PyTorch, calculating the gradient is done by using a method called `backward`. Mean squared error was just a simple standard mathematical function. PyTorch keeps track of how it was calculated and lets us calculate the derivative. So if you do a mathematical operation on a tensor in PyTorch, you can call `backward` to calculate the derivative and the derivative gets stuck inside an attribute called `.grad`.

- Take my coefficients and I'm going to subtract from them my gradient (`sub_`). There is an underscore there because that's going to do it in-place. It's going to actually update those coefficients `a` to subtract the gradients from them. Why do we subtract? Because the gradient tells us if I move the whole thing downwards, the loss goes up. If I move the whole thing upwards, the loss goes down. So I want to do the opposite of the thing that makes it go up. We want our loss to be small. That's why we subtract.

- `lr` is our learning rate. All it is is the thing that we multiply by the gradient. Why is there any `lr` at all? Let me show you why.

#### Why is there any LR at all? [[1:41:31](https://youtu.be/Egp4Zajhzog?t=6091)]

Let's take a really simple example, a quadratic. And let's say your algorithm's job was to find where that quadratic was at its lowest point. How could it do this? Just like what we're doing now, the starting point would be just to pick some x value at random. Then find out what the value of y is. That's the starting point. Then it can calculate the gradient and the gradient is simply the slope, but it tells you moving in which direction is make you go down. So the gradient tells you, you have to go this way.
![](../lesson2/lr.gif)

- If the gradient was really big, you might jump left a very long way, so you might jump all the way over to here. If you jumped over there, then that's actually not going to be very helpful because it's worse. We jumped too far so we don't want to jump too far.

- Maybe we should just jump a little bit. That is actually a little bit closer. So then we'll just do another little jump. See what the gradient is and do another little jump, and repeat.

- In other words, we find our gradient to tell us what direction to go and if we have to go a long way or not too far. But then we multiply it by some number less than 1 so we don't jump too far.

Hopefully at this point, this might be reminding you of something which is what happened when our learning rate was too high.

![](../lesson2/38.png)

Do you see why that happened now? Our learning rate was too high meant that we jumped all the way past the right answer further than we started with, and it got worse, and worse, and worse. So that's what a learning rate too high does.

On the other hand, if our learning rate is too low, then you just take tiny little steps and so eventually you're going to get there, but you are doing lots and lots of calculations along the way. So you really want to find something where it's either big enough steps like stairs or a little bit of back and forth. You want something that gets in there quickly but not so quickly it jumps out and diverges, not so slowly that it takes lots of steps. That's why we need a good learning rate and that's all it does.

So if you look inside the source code of any deep learning library, you'll find this:

 `a.sub_(lr * a.grad)`

You will find something that says "coefficients ﹣ learning rate times gradient". And we will learn about some easy but important optimization we can do to make this go faster.

That's about it. There's a couple of other little minor issues that we don't need to talk about now: one involving zeroing out the gradient and other involving making sure that you turn gradient calculation off when you do the SGD update. If you are interested, we can discuss them on the forum or you can do our introduction to machine learning course which covers all the mechanics of this in more detail.

#### Training loop [[1:45:43](https://youtu.be/Egp4Zajhzog?t=6343)]

If we run `update` 100 times printing out the loss from time to time, you can see it starts at  8.9, and it goes down.

```python
lr = 1e-1
for t in range(100): update()
```

```
tensor(8.8945, grad_fn=<MeanBackward1>)
tensor(1.6115, grad_fn=<MeanBackward1>)
tensor(0.5759, grad_fn=<MeanBackward1>)
tensor(0.2435, grad_fn=<MeanBackward1>)
tensor(0.1356, grad_fn=<MeanBackward1>)
tensor(0.1006, grad_fn=<MeanBackward1>)
tensor(0.0892, grad_fn=<MeanBackward1>)
tensor(0.0855, grad_fn=<MeanBackward1>)
tensor(0.0843, grad_fn=<MeanBackward1>)
tensor(0.0839, grad_fn=<MeanBackward1>)
```

So you can then print out scatterplots and there it is.  

```python
plt.scatter(x[:,0],y)
plt.scatter(x[:,0],x@a);
```

![](../lesson2/n3.png)

 That's it! Believe it or not, that's gradient descent. So we just need to start with a function that's a bit more complex than `x@a` but as long as we have a function that can represent things like if this is a teddy bear, we now have a way to fit it.

#### Animate it! [[1:46:20](https://youtu.be/Egp4Zajhzog?t=6380)]

Let's now take a look at this as an animation. This is one of the nice things that you can do with matplotlib. You can take any plot and turn it into an animation. So you can now actually see it updating each step.


```python
from matplotlib import animation, rc
rc('animation', html='html5')
```

> You may need to uncomment the following to install the necessary plugin the first time you run this:
> (after you run following commands, make sure to restart the kernal for this notebook)
> If you are running in colab, the installs are not needed; just change the cell above to be ... html='jshtml' instead of ... html='html5'

```python
#! sudo add-apt-repository -y ppa:mc3man/trusty-media  
#! sudo apt-get update -y
#! sudo apt-get install -y ffmpeg  
#! sudo apt-get install -y frei0r-plugins
```

Let's see what we did here. We simply said, as before, create a scatter plot, but then rather than having a loop, we used matplotlib's `FuncAnimation` to call 100 times this `animate` function.  And this function just calls that `update` we created earlier then update the `y` data in our line. Repeat that 100 times, waiting 20 milliseconds after each one.  

```python
a = nn.Parameter(tensor(-1.,1))

fig = plt.figure()
plt.scatter(x[:,0], y, c='orange')
line, = plt.plot(x[:,0], x@a)
plt.close()

def animate(i):
    update()
    line.set_ydata(x@a)
    return line,

animation.FuncAnimation(fig, animate, np.arange(0, 100), interval=20)
```

![](../lesson2/download.mp4)

You might think visualizing your algorithms with animations is something amazing and complex thing to do, but actually now you know it's 11 lines of code. So I think it's pretty darn cool.

That is SGD visualized and we can't visualize as conveniently what updating 50 million parameters in a ResNet 34 looks like but basically doing the same thing. So studying these simple version is actually a great way to get an intuition. So you should try running this notebook with a really big learning rate, with a really small learning rate, and see what this animation looks like, and try to get a feel for it. Maybe you can even try a 3D plot. I haven't tried that yet, but I'm sure it would work fine.

#### Mini-batches [[1:48:08](https://youtu.be/Egp4Zajhzog?t=6488)]

The only difference between stochastic gradient descent and this is something called _mini-batches_. You'll see, what we did here was we calculated the value of the loss on the whole dataset on every iteration. But if your dataset is 1.5 million images in ImageNet, that's going to be really slow. Just to do a single update of your parameters, you've got to calculate the loss on 1.5 million images. You wouldn't want to do that. So what we do is we grab 64 images or so at a time at random, and we calculate the loss on those 64 images, and we update our weights. Then we have another 64 random images, and we update our weights. In other words, the loop basically looks exactly the same but add some random indexes on our `x` and `y` to do a mini-batch at a time, and that would be the basic difference.

![](../lesson2/39.png)

Once you add those grab a random few points each time, those random few points are called your mini-batch, and that approach is called SGD for Stochastic Gradient Descent.

#### Vocabulary [[1:49:40](https://youtu.be/Egp4Zajhzog?t=6580)]

There's quite a bit of vocab we've just covered, so let's remind ourselves.

- **Learning rate**: A thing we multiply our gradient by to decide how much to update the weights by.

- **Epoch**: One complete run through all of our data points (e.g. all of our images). So for non-stochastic gradient descent we just did, every single loop, we did the entire dataset. But if you've got a dataset with a thousand images and our mini-batch size is 100, then it would take you 10 iterations to see every image once. So that would be one epoch. Epochs are important because if you do lots of epochs, then you are looking at your images lots of times, so every time you see an image, there's a bigger chance of overfitting. So we generally don't want to do too many epochs.

- **Mini-batch**: A random bunch of points that you use to update your weights.

- **SGD**: Gradient descent using mini-batches.

- **Model / Architecture**: They kind of mean the same thing. In this case, our architecture is <img src="http://latex.codecogs.com/gif.latex?\vec{y}&space;=&space;X\vec{a}" title="\vec{y} = X\vec{a}" />﹣ the architecture is the mathematical function that you're fitting the parameters to. And we're going to learn later today or next week what the mathematical function of things like ResNet34 actually is. But it's basically pretty much what you've just seen. It's a bunch of matrix products.

- **Parameters / Coefficients / Weights**: Numbers that you are updating.

- **Loss function**: The thing that's telling you how far away or how close you are to the correct answer. For classification problems, we use *cross entropy loss*, also known as *negative log likelihood loss*. This penalizes incorrect confident predictions, and correct unconfident predictions.


[1:51:45](https://youtu.be/Egp4Zajhzog?t=6705)

These models / predictors / teddy bear classifiers are functions that take pixel values and return probabilities. They start with some functional form like <img src="http://latex.codecogs.com/gif.latex?\vec{y}&space;=&space;X\vec{a}" title="\vec{y} = X\vec{a}" /> and they fit the parameter `a` using SGD to try and do the best to calculate your predictions. So far, we've learned how to do regression which is a single number. Next we'll learn how to do the same thing for classification where we have multiple numbers, but basically the same.

In the process, we had to do some math. We had to do some linear algebra and calculus and a lot of people get a bit scared at that point and tell us "I am not a math person". If that's you, that's totally okay. But you are wrong. You are a math person. In fact, it turns out that in the actual academic research around this, there are not "math people" and "non-math people". It turns out to be entirely a result of culture and expectations. So you should check out Rachel's talk:

[There is no such thing as "not a math person"](https://www.youtube.com/watch?v=q6DGVGJ1WP4)

![](../lesson2/rachel.png)

She will introduce you to some of that academic research. If you think of yourself as not a math person, you should watch this so that you learn that you're wrong that your thoughts are actually there because somebody has told you you're not a math person. But there's actually no academic research to suggest that there is such a thing. In fact, there are some cultures like Romania and China where the "not a math person" concept never even appeared. It's almost unheard of in some cultures for somebody to say I'm not a math person because that just never entered that cultural identity.

So don't freak out if words like derivative, gradient, and matrix product are things that you're kind of scared of. It's something you can learn. Something you'll be okay with.

#### Underfitting and Overfitting [[1:54:42](https://youtu.be/Egp4Zajhzog?t=6882)]

The last thing I want to close with is the idea of underfitting and overfitting. We just fit a line to our data. But imagine that our data wasn't actually line shaped. So if we try to fit which was something like constant + constant times X (i.e. a line) to it, it's never going to fit very well. No matter how much we change these two coefficients, it's never going to get really close.

![](../lesson2/40.png)

On the other hand, we could fit some much bigger equation, so in this case it's a higher degree polynomial with lots of wiggly bits. But if we did that, it's very unlikely we go and look at some other place to find out the temperature and how much ice cream they are selling and we will get a good result. Because the wiggles are far too wiggly. So this is called overfitting.

We are looking for some mathematical function that fits just right to stay with a teddy bear analogies. You might think if you have a statistics background, the way to make things it just right is to have exactly the same number of parameters (i.e. to use a mathematical function that doesn't have too many parameters in it). It turns out that's actually completely not the right way to think about it.

### Regularization and Validation Set [[1:56:07](https://youtu.be/Egp4Zajhzog?t=6967)]

There are other ways to make sure that we don't overfit. In general, this is called regularization. Regularization or all the techniques to make sure when we train our model that it's going to work not only well on the data it's seen but on the data it hasn't seen yet. The most important thing to know when you've trained a model is actually how well does it work on data that it hasn't been trained with. As we're going to learn a lot about next week, that's why we have this thing called a validation set.

What happens with the validation set is that we do our mini-batch SGD training loop with one set of data with one set of teddy bears, grizzlies, black bears. Then when we're done, we check the loss function and the accuracy to see how good is it on a bunch of images which were not included in the training. So if we do that, then if we have something which is too wiggly, it will tell us. "Oh, your loss function and your error is really bad because on the bears that it hasn't been trained with, the wiggly bits are in the wrong spot." Or else if it was underfitting, it would also tell us that your validation set is really bad.

Even for people that don't go through this course and don't learn about the details of deep learning, if you've got managers or colleagues at work who are wanting to learn about AI, the only thing that you really need to be teaching them is about the idea of a validation set. Because that's the thing they can then use to figure out if somebody's telling them snake oil or not. They hold back some data and they get told "oh, here's a model that we're going to roll out" and then you say "okay, fine. I'm just going to check it on this held out data to see whether it generalizes." There's a lot of details to get right when you design your validation set. We will talk about them briefly next week, but a more full version would be in Rachel's piece on the fast.ai blog called [How (and why) to create a good validation set](https://www.fast.ai/2017/11/13/validation-sets/). And this is also one of the things we go into in a lot of detail in the intro to machine learning course. So we're going to try and give you enough to get by for this course, but it is certainly something that's worth deeper study as well.

Thanks everybody! I hope you have a great time building your web applications. See you next week.
