"""
Part 2: Performance Comparisons

In this part, we will explore comparing the performance
of different pipelines.
First, we will set up some helper classes.
Then we will do a few comparisons
between two or more versions of a pipeline
to report which one is faster.
"""

import part1
import matplotlib.pyplot as plt
import time
import pandas as pd
import os

"""
=== Questions 1-5: Throughput and Latency Helpers ===

We will design and fill out two helper classes.

The first is a helper class for throughput (Q1).
The class is created by adding a series of pipelines
(via .add_pipeline(name, size, func))
where name is a title describing the pipeline,
size is the number of elements in the input dataset for the pipeline,
and func is a function that can be run on zero arguments
which runs the pipeline (like def f()).

The second is a similar helper class for latency (Q3).

1. Throughput helper class

Fill in the add_pipeline, eval_throughput, and generate_plot functions below.
"""

# Number of times to run each pipeline in the following results.
# You may modify this as you go through the file if you like, but make sure
# you set it back to 10 at the end before you submit.
NUM_RUNS = 10

class ThroughputHelper:
    def __init__(self):
        # Initialize the object.
        # Pipelines: a list of functions, where each function
        # can be run on no arguments.
        # (like: def f(): ... )
        self.pipelines = []

        # Pipeline names
        # A list of names for each pipeline
        self.names = []

        # Pipeline input sizes
        self.sizes = []

        # Pipeline throughputs
        # This is set to None, but will be set to a list after throughputs
        # are calculated.
        self.throughputs = None

    def add_pipeline(self, name, size, func):
        self.names.append(name)
        self.sizes.append(size)
        self.pipelines.append(func)

    def compare_throughput(self):
        # Measure the throughput of all pipelines
        # and store it in a list in self.throughputs.
        # Make sure to use the NUM_RUNS variable.
        # Also, return the resulting list of throughputs,
        # in **number of items per second.**
        throughputs = []

        for name, size, func in zip(self.names, self.sizes, self.pipelines):
            start = time.time()
            for _ in range(NUM_RUNS):
                func()
            end = time.time()

            elapsed = end - start
            if elapsed == 0:
                throughput = 0
            else:
                throughput = (size * NUM_RUNS) / elapsed

            throughputs.append(throughput)

        self.throughputs = throughputs
        return throughputs

    def generate_plot(self, filename):
        # Generate a plot for throughput using matplotlib.
        # You can use any plot you like, but a bar chart probably makes
        # the most sense.
        # Make sure you include a legend.
        # Save the result in the filename provided.
        if self.throughputs is None:
            raise ValueError("Run compare_throughput() before plotting.")

        plt.figure()
        plt.bar(self.names, self.throughputs, label="Throughput (items/sec)")
        plt.title("Pipeline Throughput Comparison")
        plt.xlabel("Pipeline")
        plt.ylabel("Items per Second")
        plt.legend()
        plt.xticks(rotation=30, ha='right') # rotate x labels so they don't overlap 
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

"""
As your answer to this part,
return the name of the method you decided to use in
matplotlib.

(Example: "boxplot" or "scatter")
"""

def q1():
    # Return plot method (as a string) from matplotlib
    return "bar chart"

"""
2. A simple test case

To make sure your monitor is working, test it on a very simple
pipeline that adds up the total of all elements in a list.

We will compare three versions of the pipeline depending on the
input size.
"""

LIST_SMALL = [10] * 100
LIST_MEDIUM = [10] * 100_000
LIST_LARGE = [10] * 100_000_000

def add_list(l):
    # TODO
    # Please use a for loop (not a built-in)
    total = 0
    for x in l:
        total += x
    return total

def q2a():
    # Create a ThroughputHelper object
    h = ThroughputHelper()

    # Add the 3 pipelines.
    # (You will need to create a pipeline for each one.)
    # Pipeline names: small, medium, large
    h.add_pipeline("small", len(LIST_SMALL), lambda: add_list(LIST_SMALL))
    h.add_pipeline("medium", len(LIST_MEDIUM), lambda: add_list(LIST_MEDIUM))
    h.add_pipeline("large", len(LIST_LARGE), lambda: add_list(LIST_LARGE))
    
    # Generate a plot.
    # Save the plot as 'output/part2-q2a.png'.
    # TODO
    throughputs = h.compare_throughput()
    h.generate_plot("output/part2-q2a.png")

    # Finally, return the throughputs as a list.
    # TODO
    return throughputs

"""
2b.
Which pipeline has the highest throughput?
Is this what you expected?

=== ANSWER Q2b BELOW ===
The large pipeline has the highest throughput followed by the medium, and then the small dataset. I would actually expect the smaller dataset to run faster, so this result is unexpected.

=== END OF Q2b ANSWER ===
"""

"""
3. Latency helper class.

Now we will create a similar helper class for latency.

The helper should assume a pipeline that only has *one* element
in the input dataset.

It should use the NUM_RUNS variable as with throughput.
"""

class LatencyHelper:
    def __init__(self):
        # Initialize the object.
        # Pipelines: a list of functions, where each function
        # can be run on no arguments.
        # (like: def f(): ... )
        self.pipelines = []

        # Pipeline names
        # A list of names for each pipeline
        self.names = []

        # Pipeline latencies
        # This is set to None, but will be set to a list after latencies
        # are calculated.
        self.latencies = None

    def add_pipeline(self, name, func):
        self.names.append(name) # add pipeline name and function to lists
        self.pipelines.append(func)

    def compare_latency(self):
        # Measure the latency of all pipelines
        # and store it in a list in self.latencies.
       # Also, return the resulting list of latencies in **milliseconds.**
        latencies = []
        for func in self.pipelines:
            start = time.time()
            for _ in range(NUM_RUNS):
                func()
            end = time.time()
            avg_latency = ((end - start) / NUM_RUNS) * 1000  # convert seconds → ms
            latencies.append(avg_latency)

        self.latencies = latencies
        return latencies

    def generate_plot(self, filename):
        # Generate a plot for latency using matplotlib.
        # You can use any plot you like, but a bar chart probably makes
        # the most sense.
        # Make sure you include a legend.
        # Save the result in the filename provided.
        if self.latencies is None:
            raise ValueError("Run compare_latency() before plotting.")

        plt.bar(self.names, self.latencies, label="Latency (ms)") 
        plt.title("Pipeline Latency Comparison")
        plt.xlabel("Pipeline")
        plt.ylabel("Latency (ms)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

"""
As your answer to this part,
return the number of input items that each pipeline should
process if the class is used correctly.
"""

def q3():
    # Return the number of input items in each dataset,
    # for the latency helper to run correctly.
    return 1

"""
4. To make sure your monitor is working, test it on
the simple pipeline from Q2.

For latency, all three pipelines would only process
one item. Therefore instead of using
LIST_SMALL, LIST_MEDIUM, and LIST_LARGE,
for this question run the same pipeline three times
on a single list item.
"""

LIST_SINGLE_ITEM = [10] # Note: a list with only 1 item

def q4a():
    # Create a LatencyHelper object
    h = LatencyHelper()
    # Add the single pipeline three times.
    h.add_pipeline("run1", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline("run2", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline("run3", lambda: add_list(LIST_SINGLE_ITEM))
   
    # Generate a plot.
    # Save the plot as 'output/part2-q4a.png'.
    # TODO
    latencies = h.compare_latency()
    h.generate_plot("output/part2-q4a.png")
    
    # Finally, return the latencies as a list.
    # TODO
    return latencies

"""
4b.
How much did the latency vary between the three copies of the pipeline?
Is this more or less than what you expected?

=== ANSWER Q4b BELOW ===
The latency varied slightly between the three runs. Each run got a little faster. I think the small variation is expected because is expected because all pipelines perform the same simple operation.
=== END OF Q4b ANSWER ===
"""

"""
Now that we have our helpers, let's do a simple comparison.

NOTE: you may add other helper functions that you may find useful
as you go through this file.

5. Comparison on Part 1

Finally, use the helpers above to calculate the throughput and latency
of the pipeline in part 1.
"""

# You will need these:
part1.load_input
part1.PART_1_PIPELINE

def q5a():
    # Return the throughput of the pipeline in part 1.
    h = ThroughputHelper()
    h.add_pipeline("part1_pipeline", 1, lambda: part1.PART_1_PIPELINE())
    throughputs = h.compare_throughput()
    return throughputs[0]

def q5b():
    # Return the latency of the pipeline in part 1.
    h = LatencyHelper()
    h.add_pipeline("part1_pipeline", lambda: part1.PART_1_PIPELINE())
    latencies = h.compare_latency()
    return latencies[0]

"""
===== Questions 6-10: Performance Comparison 1 =====

For our first performance comparison,
let's look at the cost of getting input from a file, vs. in an existing DataFrame.

6. We will use the same population dataset
that we used in lecture 3.

Load the data using load_input() given the file name.

- Make sure that you clean the data by removing
  continents and world data!
  (World data is listed under OWID_WRL)

Then, set up a simple pipeline that computes summary statistics
for the following:

- *Year over year increase* in population, per country

    (min, median, max, mean, and standard deviation)

How you should compute this:

- For each country, we need the maximum year and the minimum year
in the data. We should divide the population difference
over this time by the length of the time period.

- Make sure you throw out the cases where there is only one year
(if any).

- We should at this point have one data point per country.

- Finally, as your answer, return a list of the:
    min, median, max, mean, and standard deviation
  of the data.

Hints:
You can use the describe() function in Pandas to get these statistics.
You should be able to do something like
df.describe().loc["min"]["colum_name"]

to get a specific value from the describe() function.

You shouldn't use any for loops.
See if you can compute this using Pandas functions only.
"""

def load_input(filename):
    # Return a dataframe containing the population data
    # **Clean the data here**
    df = pd.read_csv(filename)

    # remove continents and world data
    df.columns = ["entity", "code", "year", "population"]
    df = df[~df["code"].str.startswith("OWID", na=False)]
    df = df[df["entity"].str.lower() != "world"]

    # drop rows with missing values
    df = df.dropna(subset=["year", "population"])

    return df

def population_pipeline(df):
    # Input: the dataframe from load_input()
    # Return a list of min, median, max, mean, and standard deviation
    grouped = df.groupby("entity")

    result = grouped.apply(
    lambda g: (g["population"].max() - g["population"].min())
              / (g["year"].max() - g["year"].min())
    if g["year"].nunique() > 1 else None,
    include_groups=False
)


    # drop na values
    result = result.dropna()

    # summary stats
    desc = result.describe()

    # Return a list of min, median, max, mean, and standard deviation
    return [
        float(desc["min"]),
        float(desc["50%"]),
        float(desc["max"]),
        float(desc["mean"]),
        float(desc["std"])
    ]

def q6():
    # As your answer to this part,
    # call load_input() and then population_pipeline()
    # Return a list of min, median, max, mean, and standard deviation
    df = load_input("data/population.csv")
    return population_pipeline(df)

"""
7. Varying the input size

Next we want to set up three different datasets of different sizes.

Create three new files,
    - data/population-small.csv
      with the first 600 rows
    - data/population-medium.csv
      with the first 6000 rows
    - data/population-single-row.csv
      with only the first row
      (for calculating latency)

You can edit the csv file directly to extract the first rows
(remember to also include the header row)
and save a new file.

Make four versions of load input that load your datasets.
(The large one should use the full population dataset.)
Each should return a dataframe.

The input CSV file will have 600 rows, but the DataFrame (after your cleaning) may have less than that.
"""
# create the files
if not os.path.exists("data/population-small.csv"):
    df = pd.read_csv("data/population.csv")
    df.head(600).to_csv("data/population-small.csv", index=False)
    df.head(6000).to_csv("data/population-medium.csv", index=False)
    df.head(1).to_csv("data/population-single-row.csv", index=False)

# first 600 rows
def load_input_small():
    df = load_input("data/population-small.csv")
    return df

# first 6000 rows
def load_input_medium():
    df = load_input("data/population-medium.csv")
    return df

# full population dataset
def load_input_large():
    df = load_input("data/population.csv") 
    return df

def load_input_single_row():
    # This is the pipeline we will use for latency.
    df = load_input("data/population-single-row.csv")
    return df

def q7():
    # Don't modify this part
    s = load_input_small()
    m = load_input_medium()
    l = load_input_large()
    x = load_input_single_row()
    return [len(s), len(m), len(l), len(x)]

"""
8.
Create baseline pipelines

First let's create our baseline pipelines.
Create four pipelines,
    baseline_small
    baseline_medium
    baseline_large
    baseline_latency

based on the three datasets above.
Each should call your population_pipeline from Q6.

Your baseline_latency function will not be very interesting
as the pipeline does not produce any meaningful output on a single row!
You may choose to instead run an example with two rows,
or you may fill in this function in any other way that you choose
that you think is meaningful.
"""

def baseline_small():
    df = load_input_small()
    return population_pipeline(df)

def baseline_medium():
    df = load_input_medium()
    return population_pipeline(df)

def baseline_large():
    df = load_input_large()
    return population_pipeline(df)

def baseline_latency():
    df = load_input_single_row() #not meaningful
    if len(df) < 2: #If not meaningful, just take 2 rows from the small dataset
        df = load_input_small().head(2)
    return population_pipeline(df)

def q8():
    # Don't modify this part
    _ = baseline_medium()
    return ["baseline_small", "baseline_medium", "baseline_large", "baseline_latency"]

"""
9.
Finally, let's compare whether loading an input from file is faster or slower
than getting it from an existing Pandas dataframe variable.

Create four new dataframes (constant global variables)
directly in the script.
Then use these to write 3 new pipelines:
    fromvar_small
    fromvar_medium
    fromvar_large
    fromvar_latency

These pipelines should produce the same answers as in Q8.

As your answer to this part;
a. Generate a plot in output/part2-q9a.png of the throughputs
    Return the list of 6 throughputs in this order:
    baseline_small, baseline_medium, baseline_large, fromvar_small, fromvar_medium, fromvar_large
b. Generate a plot in output/part2-q9b.png of the latencies
    Return the list of 2 latencies in this order:
    baseline_latency, fromvar_latency
"""

# TODO
""


POPULATION_SMALL = load_input_small()
POPULATION_MEDIUM = load_input_medium()
POPULATION_LARGE = load_input_large()
POPULATION_SINGLE_ROW = load_input_single_row()


def fromvar_small():
    return population_pipeline(POPULATION_SMALL)

def fromvar_medium():
    return population_pipeline(POPULATION_MEDIUM)

def fromvar_large():
    return population_pipeline(POPULATION_LARGE)

def fromvar_latency():
    df = POPULATION_SINGLE_ROW
    if len(df) < 2:
        df = POPULATION_SMALL.head(2)
    return population_pipeline(df)

def q9a():
    # Add all 6 pipelines for a throughput comparison
    h = ThroughputHelper()
    h.add_pipeline("baseline_small", len(POPULATION_SMALL), baseline_small)
    h.add_pipeline("baseline_medium", len(POPULATION_MEDIUM), baseline_medium)
    h.add_pipeline("baseline_large", len(POPULATION_LARGE), baseline_large)
    h.add_pipeline("fromvar_small", len(POPULATION_SMALL), fromvar_small)
    h.add_pipeline("fromvar_medium", len(POPULATION_MEDIUM), fromvar_medium)
    h.add_pipeline("fromvar_large", len(POPULATION_LARGE), fromvar_large)

    # Generate plot in ouptut/q9a.png
    throughputs = h.compare_throughput()
    h.generate_plot("output/part2-q9a.png")

    # Return list of 6 throughputs
    return throughputs

def q9b():
    # Add 2 pipelines for a latency comparison
    h = LatencyHelper()
    h.add_pipeline("baseline_latency", baseline_latency)
    h.add_pipeline("fromvar_latency", fromvar_latency)

    # Generate plot in ouptut/q9b.png
    latencies = h.compare_latency()
    h.generate_plot("output/part2-q9b.png")

    # Return list of 2 latencies
    return latencies
"""
10.
Comment on the plots above!
How dramatic is the difference between the two pipelines?
Which differs more, throughput or latency?
What does this experiment show?

===== ANSWER Q10 BELOW =====
The fromvar pipelines are consistently faster and acheieve higher throuhghput than the baseline pipelines. Throughput also changes more than latency. The throuhgput for the fromvar pipelines is about 2-3x higher than the baseline ones while the latency difference is only a few milliseconds. The fromvar pipelines run faster since memory access is quicker than file reads. 

===== END OF Q10 ANSWER =====
"""

"""
===== Questions 11-14: Performance Comparison 2 =====

Our second performance comparison will explore vectorization.

Operations in Pandas use Numpy arrays and vectorization to enable
fast operations.
In particular, they are often much faster than using for loops.

Let's explore whether this is true!

11.
First, we need to set up our pipelines for comparison as before.

We already have the baseline pipelines from Q8,
so let's just set up a comparison pipeline
which uses a for loop to calculate the same statistics.

Your pipeline should produce the same answers as in Q6 and Q8.

Create a new pipeline:
- Iterate through the dataframe entries. You can assume they are sorted.
- Manually compute the minimum and maximum year for each country.
- Compute the same answers as in Q6.
- Manually compute the summary statistics for the resulting list (min, median, max, mean, and standard deviation).
"""

def for_loop_pipeline(df):
    # Input: the dataframe from load_input()
    # Return a list of min, median, max, mean, and standard deviation
    
    result = {}
    for _, row in df.iterrows():
        country = row["entity"]
        year = row["year"]
        pop = row["population"]

        if country not in result:
            result[country] = {"min_year": year, "max_year": year,
                               "min_pop": pop, "max_pop": pop}
        else:
            if year < result[country]["min_year"]:
                result[country]["min_year"] = year
                result[country]["min_pop"] = pop
            if year > result[country]["max_year"]:
                result[country]["max_year"] = year
                result[country]["max_pop"] = pop

    # Manually compute the minimum and maximum year for each country.
    rates = []
    for country, v in result.items():
        if v["max_year"] != v["min_year"]:
            rate = (v["max_pop"] - v["min_pop"]) / (v["max_year"] - v["min_year"])
            rates.append(rate)

    rates_series = pd.Series(rates)
    stats = [
        rates_series.min(),
        rates_series.median(),
        rates_series.max(),
        rates_series.mean(),
        rates_series.std()
    ]
    return [float(x) for x in stats]

def q11():
    # As your answer to this part, call load_input() and then
    # for_loop_pipeline() to return the 5 numbers.
    # (these should match the numbers you got in Q6.)
    df = load_input("data/population.csv")
    return for_loop_pipeline(df)

"""
12.
Now, let's create our pipelines for comparison.

As before, write 4 pipelines based on the datasets from Q7.
"""

def for_loop_small():
    df = load_input_small()
    return for_loop_pipeline(df)

def for_loop_medium():
    df = load_input_medium()
    return for_loop_pipeline(df)

def for_loop_large():
    df = load_input_large()
    return for_loop_pipeline(df)

def for_loop_latency():
    df = load_input_single_row()

def q12():
    # Don't modify this part
    _ = for_loop_medium()
    return ["for_loop_small", "for_loop_medium", "for_loop_large", "for_loop_latency"]

"""
13.
Finally, let's compare our two pipelines,
as we did in Q9.

a. Generate a plot in output/part2-q13a.png of the throughputs
    Return the list of 6 throughputs in this order:
    baseline_small, baseline_medium, baseline_large, for_loop_small, for_loop_medium, for_loop_large

b. Generate a plot in output/part2-q13b.png of the latencies
    Return the list of 2 latencies in this order:
    baseline_latency, for_loop_latency
"""

def q13a():
    # Add all 6 pipelines for a throughput comparison
    h = ThroughputHelper()
    h.add_pipeline("baseline_small", len(POPULATION_SMALL), baseline_small)
    h.add_pipeline("baseline_medium", len(POPULATION_MEDIUM), baseline_medium)
    h.add_pipeline("baseline_large", len(POPULATION_LARGE), baseline_large)
    h.add_pipeline("for_loop_small", len(POPULATION_SMALL), for_loop_small)
    h.add_pipeline("for_loop_medium", len(POPULATION_MEDIUM), for_loop_medium)
    h.add_pipeline("for_loop_large", len(POPULATION_LARGE), for_loop_large)
    throughputs = h.compare_throughput()

    # Generate plot in ouptut/q13a.png
    h.generate_plot("output/part2-q13a.png")

    # Return list of 6 throughputs
    return throughputs

def q13b():
    # Add 2 pipelines for a latency comparison
    h = LatencyHelper()
    h.add_pipeline("baseline_latency", baseline_latency)
    h.add_pipeline("for_loop_latency", for_loop_latency)
    latencies = h.compare_latency()

    # Generate plot in ouptut/q13b.png
    h.generate_plot("output/part2-q13b.png")
    
    # Return list of 2 latencies
    return latencies

"""
14.
Comment on the results you got!

14a. Which pipelines is faster in terms of throughput?

===== ANSWER Q14a BELOW =====
The baseline pipelines are much faster in terms of throughput.
===== END OF Q14a ANSWER =====

14b. Which pipeline is faster in terms of latency?

===== ANSWER Q14b BELOW =====
The for-loop pipeline is slightly faster in latency, but the difference is small.
===== END OF Q14b ANSWER =====

14c. Do you notice any other interesting observations?
What does this experiment show?

===== ANSWER Q14c BELOW =====
One interesting I noticed is that the vectorized operations handled bigger datasets much more efficiently, while loops got slower as the data grows.
===== END OF Q14c ANSWER =====
"""

"""
===== Questions 15-17: Reflection Questions =====
15.

Take a look at all your pipelines above.
Which factor that we tested (file vs. variable, vectorized vs. for loop)
had the biggest impact on performance?

===== ANSWER Q15 BELOW =====
The vectorized vs. for-loop difference had the biggest impact on performance. Vectorized pipelines were way faster, while reading from a file vs. variable made only a small difference.
===== END OF Q15 ANSWER =====

16.
Based on all of your plots, form a hypothesis as to how throughput
varies with the size of the input dataset.

(Any hypothesis is OK as long as it is supported by your data!
This is an open ended question.)

===== ANSWER Q16 BELOW =====
As the dataset size increases, throughput also increases since larger batches make better use of system resources
===== END OF Q16 ANSWER =====

17.
Based on all of your plots, form a hypothesis as to how
throughput is related to latency.

(Any hypothesis is OK as long as it is supported by your data!
This is an open ended question.)

===== ANSWER Q17 BELOW =====
In general, higher throughput corresponds to lower latency for the same operation. Pipelines that can handle more data per second also tend to complete each task faster.
===== END OF Q17 ANSWER =====
"""

"""
===== Extra Credit =====

This part is optional.

Use your pipeline to compare something else!

Here are some ideas for what to try:
- the cost of random sampling vs. the cost of getting rows from the
  DataFrame manually
- the cost of cloning a DataFrame
- the cost of sorting a DataFrame prior to doing a computation
- the cost of using different encodings (like one-hot encoding)
  and encodings for null values
- the cost of querying via Pandas methods vs querying via SQL
  For this part: you would want to use something like
  pandasql that can run SQL queries on Pandas data frames. See:
  https://stackoverflow.com/a/45866311/2038713

As your answer to this part,
as before, return
a. the list of 6 throughputs
and
b. the list of 2 latencies.

and generate plots for each of these in the following files:
    output/part2-ec-a.png
    output/part2-ec-b.png
"""

# Extra credit (optional)

def extra_credit_a():
    raise NotImplementedError

def extra_credit_b():
    raise NotImplementedError

"""
===== Wrapping things up =====

**Don't modify this part.**

To wrap things up, we have collected
your answers and saved them to a file below.
This will be run when you run the code.
"""

ANSWER_FILE = "output/part2-answers.txt"
UNFINISHED = 0

def log_answer(name, func, *args):
    try:
        answer = func(*args)
        print(f"{name} answer: {answer}")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},{answer}\n')
            print(f"Answer saved to {ANSWER_FILE}")
    except NotImplementedError:
        print(f"Warning: {name} not implemented.")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},Not Implemented\n')
        global UNFINISHED
        UNFINISHED += 1

def PART_2_PIPELINE():
    open(ANSWER_FILE, 'w').close()

    # Q1-5
    log_answer("q1", q1)
    log_answer("q2a", q2a)
    # 2b: commentary
    log_answer("q3", q3)
    log_answer("q4a", q4a)
    # 4b: commentary
    log_answer("q5a", q5a)
    log_answer("q5b", q5b)

    # Q6-10
    log_answer("q6", q6)
    log_answer("q7", q7)
    log_answer("q8", q8)
    log_answer("q9a", q9a)
    log_answer("q9b", q9b)
    # 10: commentary

    # Q11-14
    log_answer("q11", q11)
    log_answer("q12", q12)
    log_answer("q13a", q13a)
    log_answer("q13b", q13b)
    # 14: commentary

    # 15-17: reflection
    # 15: commentary
    # 16: commentary
    # 17: commentary

    # Extra credit
    log_answer("extra credit (a)", extra_credit_a)
    log_answer("extra credit (b)", extra_credit_b)

    # Answer: return the number of questions that are not implemented
    if UNFINISHED > 0:
        print("Warning: there are unfinished questions.")

    return UNFINISHED

"""
=== END OF PART 2 ===

Main function
"""

if __name__ == '__main__':
    log_answer("PART 2", PART_2_PIPELINE)

