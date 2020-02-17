# Part-of-speech-tagging-using-HMM-and-Viterbi-algorithm
In this project, we processed POS Tagging algorithm using Hidden Markov Model (HMM) then build
the Viterbi algorithm based on POS tagger.

## POS Tagging algorithm using Hidden Markov Model (HMM)

Before we use the HMM algorithm to tag the words, it is necessary for us to do the Data Preparation such
as checking some of the tagged data, splitting the data into training and validation set in the ratio 95:5 and
creating list of train and test tagged words etc.. After that, we could start to process the POS Tagging algorithm
using HMM. Given a sequence of words to be tagged, the task is to assign the most probable tag to the word.
In other words, to every word w, assign the tag t that maximize the likelihood P(t/w). Since P(t/w) = P(w/t).
P(t) / P(w), after ignoring P(w), we have to compute P(w/t) and P(t).

## Viterbi algorithm
The steps are as follows:
1. Given a sequence of words
2. iterate through the sequence
3. for each word (starting from first word in sequence) calculate the product of emission probabilities
and transition probabilities for all possible tags.
4. assign the tag which has maximum probability obtained in step 3 above.
5. move to the next word in sequence to repeat steps 3 and 4 above.
