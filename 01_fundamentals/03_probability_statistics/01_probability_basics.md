# Contents for the file: /01_fundamentals/03_probability_statistics/01_probability_basics.md

# Probability Basics

## Introduction to Probability
Probability is a branch of mathematics that deals with the likelihood of events occurring. It provides a framework for quantifying uncertainty and making informed decisions based on incomplete information. Understanding probability is essential in machine learning, as it helps us model and predict outcomes based on data.

## Key Concepts

### 1. What is Probability?
Probability measures how likely an event is to occur, expressed as a number between 0 and 1. An event with a probability of 0 will never happen, while an event with a probability of 1 is certain to happen.

**Example**: 
- The probability of flipping a fair coin and getting heads is 0.5 (or 50%).

### 2. Types of Events
- **Independent Events**: Two events are independent if the occurrence of one does not affect the other. 
  - *Example*: Flipping a coin and rolling a die.
  
- **Dependent Events**: Two events are dependent if the occurrence of one affects the probability of the other.
  - *Example*: Drawing cards from a deck without replacement.

### 3. Conditional Probability
Conditional probability is the probability of an event occurring given that another event has already occurred. It is denoted as P(A|B), which reads "the probability of A given B."

**Formula**: 
P(A|B) = P(A ∩ B) / P(B)

**Example**: 
If you have a deck of cards, the probability of drawing an Ace (event A) given that you have drawn a Spade (event B) is calculated based on the number of Aces in the Spades.

### 4. Bayes' Theorem
Bayes' Theorem relates the conditional and marginal probabilities of random events. It allows us to update our beliefs based on new evidence.

**Formula**: 
P(A|B) = [P(B|A) * P(A)] / P(B)

**Example**: 
If a medical test for a disease is 90% accurate, Bayes' theorem can help determine the probability of having the disease given a positive test result.

## Why Does This Matter?
Understanding probability is crucial for making predictions and decisions in machine learning. It helps in:
- Evaluating the uncertainty in predictions.
- Making informed decisions based on data.
- Developing algorithms that can learn from data and improve over time.

## Practical Exercises
1. **Thought Experiment**: Consider a bag containing 3 red balls and 2 blue balls. What is the probability of drawing a red ball? What if you draw one ball and do not replace it?
2. **Real-World Application**: Think about a scenario in your life where you make decisions based on probabilities (e.g., weather forecasts, sports betting). How do you assess the likelihood of different outcomes?

## Conclusion
Probability forms the backbone of many machine learning algorithms. By mastering the basics of probability, you will be better equipped to understand more complex concepts in statistics and machine learning.

---

This file serves as an introduction to the fundamental concepts of probability, providing a solid foundation for further exploration in statistics and machine learning.