# judging-a-movie-by-its-cover
This repository houses the program documents for a CNN based genre predictor, based on movie posters.


Background:

  Though the common wisdom is to not judge a book by its cover,  more often then not this is exactly how we approach many choices in life. I don't think I'v ever bought a wine, for instance, with out first picking a price range and then the bottle with the label I thought looked most like a wine I'd enjoy.
  The same goes for movies, in many cases. Despite the masses of reviews, critiques, and categorizations we are often presented with a wall of movie posters that a service like Netflix or Hulu is guessing we may like. Recommendation engines are becoming more and more sophisticated, but when it comes to movie content many of us still follow our eyes and pick a movie based just on the poster.

Goals:

  The goal of this project is to understand the visual content of a poster using modern Machine Learning techniques. A cursory glance at some posters may make this task seem trivial. Follow this link to see some similar posters - https://imgur.com/gallery/jrmWj. As you study these,  however, you can see though the elements seem similar they are not all, in fact, of the same ilk. The "Standing Back-to-Back" section includes Buddy Comedies, Romance and Action Movie.
  I posit that a well trained and tuned Convolutional Neural Network has the dispassionate, raw computing strength to see past the obvious features and tease out the hidden layers in movie images.
  This tool could prove useful as an element of recommender engines or an aid to movie sites, like Rottontomatoes.com or Netflix, when they import and classify new content that may not include appropriate genre tags.

The Data:

  The data I'm using here are 10000 (5010 and counting) movie posters obtained from Rottontomatoes.com. They are indexed with their genres and some other information (it would be interesting, if a second iteration of this algorithm could predict rating!).
