# Experiments in machine learning and deep learning with Forex

This repo is intended as some sort of a (very disorganized, highly unacademic, and somewhat embarrassing) public archive of a year long experimentation of applying machine learning techniques to FX charts.

The idea was as follows: Can we use breakthroughs in deep learning to replicate the behavior of an experienced day trader?

I started by collecting some data from my friend Rudy's day-trading platform, including getting some export scripts in place.

Then, I tried to apply some simple feed forward networks.

Then, I tried random forests, and about everything else relevant `sklearn`. 

Then, I though "Why not try applying some breakthrough papers in image recognition to the problem of 'identifying' these shapes, but represented numerically instead ?"

Then, came a whole lot of work, specifically with convolutional nets, LSTMs, and autoencoders. Everytime I found a new paper that seemed relevant I would try to implement it for my problem. 

In the end, a lot of feature engineering (with help from my awesome tutor, Joe) and a pretty simple Conv1D architecture based loosely on a paper related to mammogram identification achieved the highest F1 score and was actually pretty useful. 

About 5% of the code I wrote is contained herein.

If you find it useful, let me know!

