import numpy as np
import matplotlib.pyplot as plt

# Sign function given in class
def sign(x):
    if x < 0:
        return -1
    else:
        return 1
    
# Generate random patterns
def generate_patterns(num_patterns, num_neurons):

    #2D array of -1's and 1's of size num_patterns x num_neurons
    return np.random.choice([-1, 1], size=(num_patterns, num_neurons))

# Function for imprinting patterns
def imprint_patterns(patterns, num_neurons):
    #Initialize the weights
    weights = np.zeros((num_neurons, num_neurons))

    # Sum from 1 to patterns (p)
    for pattern in patterns:
        #Sum of outer products
        weights += np.outer(pattern, pattern)

    # 1 / n at the end
    weights /= num_neurons

    #make sure Wij = 0 where i = j
    np.fill_diagonal(weights, 0)

    return weights

def test_stability(weights, patterns):
    num_patterns, num_neurons = patterns.shape # Get number of neurons and patterns, which should be 100 and 50, respectively
    stable_counts = np.zeros(num_patterns) # Get a count of stable patterns for each 
    for pattern in range(num_patterns):
        state = np.copy(patterns[pattern]) #get sj

        #Sum from i to N part
        for i in range(num_neurons):
            h_i = np.dot(weights[i], state) # Wij * sj
            new_state = sign(h_i) # si prime = sigma of hi
            if new_state != state[i]: #Not stable if si prime != si
                break
        else:
            stable_counts[pattern] = 1 #Pattern is stable if we get here

    return stable_counts

#For each trial, generate a pattern. For each p from 1 to 50, imprint the patterns up to p and test stability. 
def run_experiment(num_patterns=50, num_neurons=100, trials=5):
    # Get a list of unstable fractions and stable counts for each trial
    unstable_fractions = np.zeros((trials, num_patterns))
    stable_counts_all = np.zeros((trials, num_patterns))
    
    for trial in range(trials):
        patterns = generate_patterns(num_patterns, num_neurons)
        stable_counts = np.zeros(num_patterns)
        
        for p in range(1, num_patterns + 1):
            weights = imprint_patterns(patterns[:p], num_neurons)

            #Aggregate the stable counts for each p
            stable_counts[p - 1] = np.sum(test_stability(weights, patterns[:p]))
        
        #Get the stable counts and unstable fractions for each trial
        stable_counts_all[trial] = stable_counts
        # Basically 1 - (stable counts / p)
        unstable_fractions[trial] = 1 - (stable_counts / np.arange(1, num_patterns + 1))
    
    return unstable_fractions, stable_counts_all

# Plot the results of each trial
def plot_results(unstable_fractions, stable_counts):
    p_values = np.arange(1, unstable_fractions.shape[1] + 1)

    plt.figure(figsize=(12, 5))

    # Plot the fraction of unstable imprints
    plt.subplot(1, 2, 1)
    for trial in range(unstable_fractions.shape[0]):
        plt.plot(p_values, unstable_fractions[trial], label=f'Trial {trial+1}')
    plt.xlabel('Number of Imprints (p)')
    plt.ylabel('Fraction of Unstable Imprints')
    plt.title('Fraction of Unstable Imprints vs. p')
    plt.legend()

    # Plot the number of stable imprints
    plt.subplot(1, 2, 2)
    for trial in range(stable_counts.shape[0]):
        plt.plot(p_values, stable_counts[trial], label=f'Trial {trial+1}')
    plt.xlabel('Number of Imprints (p)')
    plt.ylabel('Number of Stable Imprints')
    plt.title('Number of Stable Imprints vs. p')
    plt.legend()

    plt.tight_layout()
    #Save figure at end. Let's not show it since this isn't a notebook
    plt.savefig('hopfield_experiment_results.png')

    #Do the same thing, but show an average of the trials, with a standard deviation being highlighted

    plt.figure(figsize=(12, 5))

    # Plot the fraction of unstable imprints
    plt.subplot(1, 2, 1)

    #Get the mean and standard deviation of the unstable fractions
    unstable_fractions_mean = np.mean(unstable_fractions, axis=0)
    unstable_fractions_std = np.std(unstable_fractions, axis=0)

    plt.plot(p_values, unstable_fractions_mean, label='Mean')
    plt.fill_between(p_values, unstable_fractions_mean - unstable_fractions_std, unstable_fractions_mean + unstable_fractions_std, alpha=0.3, label='Mean +/- 1 Standard Deviation')
    plt.xlabel('Number of Imprints (p)')
    plt.ylabel('Fraction of Unstable Imprints')
    plt.title('Fraction of Unstable Imprints vs. p')
    plt.legend()

    # Plot the number of stable imprints
    plt.subplot(1, 2, 2)

    #Get the mean and standard deviation of the stable counts
    stable_counts_mean = np.mean(stable_counts, axis=0)
    stable_counts_std = np.std(stable_counts, axis=0)

    plt.plot(p_values, stable_counts_mean, label='Mean')
    plt.fill_between(p_values, stable_counts_mean - stable_counts_std, stable_counts_mean + stable_counts_std, alpha=0.3, label='Mean +/- 1 Standard Deviation')
    plt.xlabel('Number of Imprints (p)')
    plt.ylabel('Number of Stable Imprints')
    plt.title('Number of Stable Imprints vs. p')
    plt.legend()

    plt.tight_layout()
    #Save figure at end. Let's not show it since this isn't a notebook
    plt.savefig('hopfield_experiment_results_mean_std.png')

# Run the experiment and plot the results
unstable_fractions, stable_counts = run_experiment()
plot_results(unstable_fractions, stable_counts)

