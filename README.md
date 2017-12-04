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
To train my model I scraped some 10,000 images from Rotten Tomatoes using the Beautiful Soup module. I applied for an API, but since RTs parent company couldn't be bothered, I implimented some simple sleep commands in my code to slowly, under the radar, pick away at the poster images hosted on their site.

## Preprocessing and Munging
Minimal preprocesssing was required. Images were of a uniform style and sorted during scraping into their classes. These presorted classes need some addressing. Rotten Tomatoes often applies more than one genre tag to a movie so it was necessary to make some decisions as to what belonged where.


For movies that listed both Children's and Comendy in the genre, I called that Comedy, for example. Horror movies are all lumped into the Action/Thriller category. To simplify matters further images were transformed into square arrays of 130px X 130px. These were run through a Keras generator that preformed some light zooming and recentering, which effectively upsampled my test data.

## Network Architecture
After examining the architectures of some proven CNN’s (ImageNet, CIFAR10, VGG) I used python, numpy, sklearn, keras and tensor flow to put together my own architecture and trained my model using the powerful processing available through Amazon’s web service. I ended up with twelve (12) Convolutional Layers activated by the ReLU (Rectified Linear Unit) function and six (6) Dense Layers also activated with ReLU.

The results are rather promising. For 3 classes of film, Action/Thriller, Comedy and Drama the accuracy was 66%, which isn’t amazing but is as we say in Maine, “ betta’ than guessin’.” I did start with 8 classes and then the accuracy was only 33%. When I moved to just five classes I was able to see and improvement to over 45%.

If there’s time I’d like to demo my app. JAM by Jack.

Going forward I’d like to further tune the existing models to eek out the best results I can. Assuming I can affect some meaningful change it would be appropriate to start looking at other aspects of a movie more related to enjoyment. i. e. Rating.


Background:

  Though the common wisdom is to not judge a book by its cover,  more often then not this is exactly how we approach many choices in life. I don't think I'v ever bought a wine, for instance, with out first picking a price range and then the bottle with the label I thought looked most like a wine I'd enjoy.
  The same goes for movies, in many cases. Despite the masses of reviews, critiques, and categorizations we are often presented with a wall of movie posters that a service like Netflix or Hulu is guessing we may like. Recommendation engines are becoming more and more sophisticated, but when it comes to movie content many of us still follow our eyes and pick a movie based just on the poster.


Goals:


  The goal of this project is to understand the visual content of a poster using modern Machine Learning techniques. A cursory glance at some posters may make this task seem trivial. Follow this link to see some similar posters - https://imgur.com/gallery/jrmWj. As you study these,  however, you can see though the elements seem similar they are not all, in fact, of the same ilk. The "Standing Back-to-Back" section includes Buddy Comedies, Romance and Action Movie.

  I posit that a well trained and tuned Convolutional Neural Network has the dispassionate, raw computing strength to see past the obvious features and tease out the hidden layers in movie images.
  This tool could prove useful as an element of recommender engines or an aid to movie sites, like Rottontomatoes.com or Netflix, when they import and classify new content that may not include appropriate genre tags.

The Data:

  The data I'm using here are 5010 and counting movie posters obtained from Rottontomatoes.com. They are indexed with their genres and some other information (it would be interesting, if a second iteration of this algorithm could predict rating!).


  
