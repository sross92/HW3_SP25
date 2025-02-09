#Utilized ChatGPT for code debugging after adding in Secant method call, built off hw2a.py program provided by Dr.Smay
#region imports
from numericalMethods import GPDF, Probability, Secant #imports functions from numerical methods to be called and used.
#endregion


#region function definitions
def find_c(mean, stDev, target_P, GT=True, OneSided=True):
    """
    Uses the Secant method to find the value of c that results in the target probability.
    """

    def error_function(c): #this calculates the difference between computed and target probability
        return Probability(GPDF, (mean, stDev), c, GT) - target_P

    c_guess1, c_guess2 = mean, mean + stDev  # Initial guesses
    c_solution, iterations = Secant(error_function, c_guess1, c_guess2)
    return c_solution

def main():
    """
    This program is designed to solicit input from the CLI and return a probability from the Gaussian Normal Distribution
    by integrating the probability density function using Secant Method for numerical integration.

    The user may decide upon a two-sided integration or one-sided integration.
    If one-sided, we integrate from the smaller of mean-5*stDev or c-stDev.
    If two-sided, we integrate from -c to c.
    Here is my step-by-step plan:
    1. Decide mean, stDev, and c or P and if I want P(x>c) or P(x<c) or P(-c<x<c) or P(-c>x>c).
    2. Define args tuple and c to be passed to Probability
    3. Pass args, and a callback function (GPDF) to Probability
    4. In probability, pass along GPDF to Secant along with the appropriate args tuple
    5. Return the required probability from Probability and print to screen.
    :return: Nothing to return, just print results to screen.
    """

#region testing user input
    # setting the initial default values
    Again = True
    mean = 0
    stDev = 1.0
    c = 0.5
    OneSided = True  # integrates from mu-5*sig if true, from mu-(c-mu) to mu+(c-mu) if False
    GT = False
    yesOptions = ["y","yes","true"]
    while Again:
        response = input(f"Population mean? ({mean:0.3f})").strip().lower() #Asks user for Mean Population
        mean = float(response) if response != '' else mean

        response = input(f"Standard deviation? ({stDev:0.3f})").strip().lower() #Asks user for Standard Deviation
        stDev = float(response) if response != '' else stDev

        response = input("Are you specifying c (to find P) or specifying P (to find c)? (c/P)").strip().lower()
        find_c_mode = response == "p"

        if find_c_mode:
            response = input("Enter desired probability P: ").strip().lower()
            target_P = float(response)
            response = input(f"Probability greater than c? ({GT})").strip().lower()
            GT = True if response in yesOptions else False
            response = input(f"One sided? ({OneSided})").strip().lower()
            OneSided = True if response in yesOptions else False
            c = find_c(mean, stDev, target_P, GT, OneSided)
            print(f"Value of c that gives probability {target_P:.3f} is: {c:.3f}")
        else:
            response = input(f"c value? ({c:0.3f})").strip().lower()
            c = float(response) if response != '' else c
            response = input(f"Probability greater than c? ({GT})").strip().lower()
            GT = True if response in yesOptions else False
            response = input(f"One sided? ({OneSided})").strip().lower()
            OneSided = True if response in yesOptions else False

            if OneSided:
                prob = Probability(GPDF, (mean, stDev), c, GT)
                print(f"P(x {'>' if GT else '<'} {c:.2f} | {mean:.2f}, {stDev:.2f}) = {prob:.3f}")
            else:
                prob = Probability(GPDF, (mean, stDev), c, GT=True)
                prob = 1 - 2 * prob
                if GT:
                    print(
                        f"P({mean - (c - mean):.2f} > x > {mean + (c - mean):.2f} | {mean:.2f}, {stDev:.2f}) = {1 - prob:.3f}")
                else:
                    print(
                        f"P({mean - (c - mean):.2f} < x < {mean + (c - mean):.2f} | {mean:.2f}, {stDev:.2f}) = {prob:.3f}")

        response = input("Go again? (Y/N)").strip().lower()
        Again = response in yesOptions


#endregion

if __name__ == "__main__":
    main()