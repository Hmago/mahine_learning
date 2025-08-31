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

## Mathematical Foundation

### Key Formulas

**Basic Probability:**
$$P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}$$

**Probability Axioms:**

1. $P(A) \geq 0$ for any event $A$
2. $P(\Omega) = 1$ where $\Omega$ is the sample space
3. For mutually exclusive events: $P(A \cup B) = P(A) + P(B)$

**Conditional Probability:**
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**Bayes' Theorem:**
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**Total Probability:**
$$P(B) = \sum_{i} P(B|A_i) \cdot P(A_i)$$

**Independence:**
Events $A$ and $B$ are independent if $P(A \cap B) = P(A) \cdot P(B)$

### Solved Examples

#### Example 1: Basic Probability Calculation

Given: A fair six-sided die is rolled twice

Find: Probability of getting sum equals 7

Solution:
Step 1: Identify sample space
Total outcomes = $6 \times 6 = 36$ possible pairs

Step 2: Count favorable outcomes
Pairs that sum to 7: $(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)$ = 6 outcomes

Step 3: Calculate probability
$$P(\text{sum} = 7) = \frac{6}{36} = \frac{1}{6} \approx 0.167$$

Result: The probability is $\frac{1}{6}$ or about 16.7%.

#### Example 2: Conditional Probability and Bayes' Theorem

Given: Medical test scenario
- Disease prevalence: $P(D) = 0.01$ (1% of population has disease)
- Test sensitivity: $P(T^+|D) = 0.95$ (95% accuracy if disease present)
- Test specificity: $P(T^-|D^c) = 0.90$ (90% accuracy if disease absent)

Find: $P(D|T^+)$ (probability of having disease given positive test)

Solution:
Step 1: Calculate $P(T^+|D^c)$
$$P(T^+|D^c) = 1 - P(T^-|D^c) = 1 - 0.90 = 0.10$$

Step 2: Calculate $P(T^+)$ using total probability
$$P(T^+) = P(T^+|D) \cdot P(D) + P(T^+|D^c) \cdot P(D^c)$$
$$P(T^+) = 0.95 \times 0.01 + 0.10 \times 0.99 = 0.0095 + 0.099 = 0.1085$$

Step 3: Apply Bayes' theorem
$$P(D|T^+) = \frac{P(T^+|D) \cdot P(D)}{P(T^+)} = \frac{0.95 \times 0.01}{0.1085} = \frac{0.0095}{0.1085} \approx 0.088$$

Result: Even with a positive test, there's only an 8.8% chance of actually having the disease!

#### Example 3: Independence vs Dependence

Given: Two cards drawn from standard deck
- Scenario A: With replacement
- Scenario B: Without replacement

Find: Probability of drawing two Aces in each scenario

Solution:
**Scenario A (With replacement):**
Step 1: First card probability
$$P(\text{Ace}_1) = \frac{4}{52} = \frac{1}{13}$$

Step 2: Second card probability (independent)
$$P(\text{Ace}_2|\text{Ace}_1) = \frac{4}{52} = \frac{1}{13}$$

Step 3: Calculate joint probability
$$P(\text{Ace}_1 \cap \text{Ace}_2) = P(\text{Ace}_1) \times P(\text{Ace}_2) = \frac{1}{13} \times \frac{1}{13} = \frac{1}{169} \approx 0.0059$$

**Scenario B (Without replacement):**
Step 1: First card probability
$$P(\text{Ace}_1) = \frac{4}{52} = \frac{1}{13}$$

Step 2: Second card probability (dependent)
$$P(\text{Ace}_2|\text{Ace}_1) = \frac{3}{51} = \frac{1}{17}$$

Step 3: Calculate joint probability
$$P(\text{Ace}_1 \cap \text{Ace}_2) = \frac{4}{52} \times \frac{3}{51} = \frac{12}{2652} = \frac{1}{221} \approx 0.0045$$

Result: Replacement affects probability due to changing dependence structure.

---

This file serves as an introduction to the fundamental concepts of probability, providing a solid foundation for further exploration in statistics and machine learning.