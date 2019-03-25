# Deployment-of-model
Developed an app to predict whether food reviews are positive or negative using linear SVM on amazon fine food review dataset and deployed it on localhost
The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon.

Number of reviews: 568,454 Number of users: 256,059 Number of products: 74,258 Timespan: Oct 1999 - Oct 2012 Number of Attributes/Columns in data: 10

Attribute Information:

Id ProductId - unique identifier for the product UserId - unqiue identifier for the user ProfileName HelpfulnessNumerator - number of users who found the review helpful HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not Score - rating between 1 and 5 Time - timestamp for the review Summary - brief summary of the review Text - text of the review Objective: Given a review, determine whether the review is positive or negative.

From this dataset first build linear SVM with BOW vectorizer.
Then, takes the pickle file of SVM model and pickle file of count vectorizer using joblib.dump() function.
After that, wrote a html programme to predict on PC using localhost (local server)
Lastly, build app.py using flask to deploy our model on production to check whether review is positive or negative.

Attached some screenshots.
