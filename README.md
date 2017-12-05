# Judging a Movie by Its Cover

This repository houses the program documents for a CNN based genre predictor, based on movie posters.

We were all taught from a young age, “Don’t judge a book by its cover.” Unfortunately that’s not generally the way we make those types of decision. When we log onto Netflix we’re presented with a poster, which many of us use as the sole factor in choosing a title to watch. I wanted to delve into this problem using machine learning and ask whether a machine can make categorical distinctions based just on poster input. I started at the top level, to assure there were in fact discernible features, which is to say if I could predict genre from a movie poster.
A tool like this could be useful to international companies interesting in loading foreign or pre-computer age titles into their systems while avoiding costly human analysis.

To tackle this problem I employed a Convolutional Neural Network (CNNs), which is a type of Neural Network designed to mimic the human visual system, quickly identifying features in visual input, such as color, brightness, vertical and horizontal lines uncover hidden connections in the visual information of a movie poster. Importantly, CNNs preserve the spatial relationships in images. Image classification using CNNs is much quicker than were humans to do the same task. A tool like this could be useful to international companies interesting in loading foreign or pre-computer age titles into their systems while avoiding costly human analysis.

### Contents:
1. Data Aquisition
2. Preprocessing and Munging
3. Network Architecture
4. iOS App Implimentation

## Data Aquisition
To train my model I scraped some 10,000 images from Rotten Tomatoes using the Beautiful Soup module. I applied for an API, but since RTs parent company couldn't be bothered, I implemented some simple sleep commands in my code to slowly, under the radar, pick away at the poster images hosted on their site.

## Preprocessing and Munging
Minimal preprocesssing was required. Images were of a uniform style and sorted during scraping into their classes. These presorted classes need some addressing. Rotten Tomatoes often applies more than one genre tag to a movie so it was necessary to make some decisions as to what belonged where.

For movies that listed both Children's and Comedy in the genre, I called that Comedy, for example. Horror movies are all lumped into the Action/Thriller category. To simplify matters further images were transformed into square arrays of 130px X 130px. These were run through a Keras generator that preformed some light zooming and centering, which effectively upsampled my test data.

## Network Architecture
After examining the architectures of some proven CNN’s (ImageNet, CIFAR10, VGG) I used python, numpy, sklearn, keras and tensor flow to put together my own architecture and trained my model using the powerful processing available through Amazon’s web service. I ended up with twelve (12) Convolutional Layers activated by the ReLU (Rectified Linear Unit) function and six (6) Dense Layers also activated with ReLU.

## Results
The results are rather promising. For 3 classes of film, Action/Thriller, Comedy and Drama the accuracy was 66%, which isn’t amazing but is as we say in Maine, “ betta’ than guessin’.” I did start with 8 classes and then the accuracy was only 33%. When I moved to just five classes I was able to see and improvement to over 45%. Recall on 3 classes was close to 70% in the Comedy genre, which I believe is due to the bright, colorful nature of many comedy movie posters. I think what confounds these results somewhat is the poster designers desire to emulate the media associated with popular films - see 'Hot Shots: Part Duex' vs. Rambo III.


## iOS App Implimentation

Using Swift and CoreML, Apple's programing language and under-the-hood Machine Learning architecture, I created an App to accept image data and make predictions based on the trained model, described above. This is a fun tool and an example of how a tool like this could be deployed.




























Goals:


  The goal of this project is to understand the visual content of a poster using modern Machine Learning techniques. A cursory glance at some posters may make this task seem trivial. Follow this link to see some similar posters - https://imgur.com/gallery/jrmWj. As you study these,  however, you can see though the elements seem similar they are not all, in fact, of the same ilk. The "Standing Back-to-Back" section includes Buddy Comedies, Romance and Action Movie.

  I posit that a well trained and tuned Convolutional Neural Network has the dispassionate, raw computing strength to see past the obvious features and tease out the hidden layers in movie images.
  This tool could prove useful as an element of recommender engines or an aid to movie sites, like Rottontomatoes.com or Netflix, when they import and classify new content that may not include appropriate genre tags.

The Data:

  The data I'm using here are 5010 and counting movie posters obtained from Rottontomatoes.com. They are indexed with their genres and some other information (it would be interesting, if a second iteration of this algorithm could predict rating!).
