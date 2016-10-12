package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;

// TODO: FIX ME SO THAT I'M NOT JUST THE PERCEPTRON! :((((

/**
 * Gradient descent classifier allowing for two different loss functions and
 * three different regularization settings.
 * 
 * @author Madhura Jayaraman and Charles Watson
 *
 */
public class GradientDescentClassifier implements Classifier {
	// constants for the different surrogate loss functions
	public static final int EXPONENTIAL_LOSS = 0;
	public static final int HINGE_LOSS = 1;

	// constants for the different regularization parameters
	public static final int NO_REGULARIZATION = 0;
	public static final int L1_REGULARIZATION = 1;
	public static final int L2_REGULARIZATION = 2;

	protected HashMap<Integer, Double> weights; // the feature weights
	protected double b = 0; // the intersect weight

	protected int iterations = 10;

	private double lambda = 0.0;
	private double eta = 0.0;
	private int whichLoss = 0;
	private int whichReg = 0;

	/**
	 * Zero parameter constructor. Its default is exponential loss, no
	 * regularization, lambda=0.1 and eta=0.1
	 *
	 */
	public GradientDescentClassifier() {
		lambda = 0.1;
		eta = 0.1;
		setLoss(EXPONENTIAL_LOSS);
		setRegularization(NO_REGULARIZATION);
	}

	/**
	 * Get a weight vector over the set of features with each weight set to 0
	 * 
	 * @param features
	 *            the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features) {
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();

		for (Integer f : features) {
			temp.put(f, 0.0);
		}

		return temp;
	}

	/**
	 * Initialize the weights and the intersect value
	 * 
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features) {
		weights = getZeroWeights(features);
		b = 0;
	}

	/**
	 * Set the number of iterations the perceptron should run during training
	 * 
	 * @param iterations
	 */
	public void setIterations(int iterations) {
		this.iterations = iterations;
	}

	/**
	 * Calculates the correct loss function given a label and feature value
	 */
	public double getLoss(double label, double dotProduct) {
		if (whichLoss == EXPONENTIAL_LOSS) {
			// implement exponential loss
			return Math.exp(-label * dotProduct);
		} else {
			// implement hinge loss
			// max(0,1−yy')
			return Math.max(0, 1 - (label * dotProduct));
		}
	}

	/**
	 * Sets the loss function to be employed when method getLoss is called
	 * 
	 * @param loss
	 *            - which loss function to use
	 */
	public void setLoss(int loss) {
		if (loss == HINGE_LOSS)
			whichLoss = HINGE_LOSS;
		else if (loss == EXPONENTIAL_LOSS)
			whichLoss = EXPONENTIAL_LOSS;
	}

	/**
	 * Calculates the correct regularization value given input weight
	 * 
	 * @param weight
	 *            - input weight value
	 */
	public double getRegularization(double weight) {
		if (whichReg == L1_REGULARIZATION) {
			return Math.signum(weight);
		} else if (whichReg == L2_REGULARIZATION) {
			return weight;
		} else {
			return 0.0;
		}
	}

	/**
	 * Sets the regularization function used when method getRegularization is
	 * called
	 * 
	 * @param reg
	 *            - indicating which regularization to use
	 */
	public void setRegularization(double reg) {
		if (reg == L1_REGULARIZATION) {
			whichReg = L1_REGULARIZATION;
		} else if (reg == L2_REGULARIZATION) {
			whichReg = L2_REGULARIZATION;
		} else if (reg == NO_REGULARIZATION) {
			whichReg = NO_REGULARIZATION;
		}
	}

	/**
	 * Calculates the dot product -- our prediction -- between the perceptron's
	 * weight vector and the example's feature values, and adds bias term
	 * 
	 * @param weights
	 *            perceptron weight vector
	 * @param bias
	 *            bias term
	 * @param example
	 *            example from our dataset
	 * @return prediction
	 */
	public double calculateDotProductWithBias(HashMap<Integer, Double> weights, double bias, Example example) {
		if (weights.size() != example.getFeatureSet().size())
			return 0.0;

		double sum = 0.0;
		for (int i = 0; i < weights.size(); i++) {
			sum += weights.get(i) * example.getFeature(i);
		}
		return sum + bias;
	}

	public void train(DataSet data) {
		initializeWeights(data.getAllFeatureIndices());

		ArrayList<Example> training = (ArrayList<Example>) data.getData().clone();

		for (int it = 0; it < iterations; it++) {
			Collections.shuffle(training);

			for (Example e : training) {
				if (getPrediction(e) != e.getLabel()) {
					double label = e.getLabel();

					// update the weights
					// for( Integer featureIndex: weights.keySet() ){
					for (Integer featureIndex : e.getFeatureSet()) {
						double oldWeight = weights.get(featureIndex);
						double featureValue = e.getFeature(featureIndex);
						double dotProduct = calculateDotProductWithBias(weights, b, e);

						weights.put(featureIndex, oldWeight + eta * label * featureValue * getLoss(label, dotProduct)
								- lambda * eta * getRegularization(oldWeight));
					}

					// update b
					b += label;
				}
			}
		}
	}

	@Override
	public double classify(Example example) {
		return getPrediction(example);
	}

	@Override
	public double confidence(Example example) {
		return Math.abs(getDistanceFromHyperplane(example, weights, b));
	}

	/**
	 * Get the prediction from the current set of weights on this example
	 * 
	 * @param e
	 *            the example to predict
	 * @return
	 */
	protected double getPrediction(Example e) {
		return getPrediction(e, weights, b);
	}

	/**
	 * Get the prediction from the on this example from using weights w and
	 * inputB
	 * 
	 * @param e
	 *            example to predict
	 * @param w
	 *            the set of weights to use
	 * @param inputB
	 *            the b value to use
	 * @return the prediction
	 */
	protected static double getPrediction(Example e, HashMap<Integer, Double> w, double inputB) {
		double sum = getDistanceFromHyperplane(e, w, inputB);

		if (sum > 0) {
			return 1.0;
		} else if (sum < 0) {
			return -1.0;
		} else {
			return 0;
		}
	}

	protected static double getDistanceFromHyperplane(Example e, HashMap<Integer, Double> w, double inputB) {
		double sum = inputB;

		// for(Integer featureIndex: w.keySet()){
		// only need to iterate over non-zero features
		for (Integer featureIndex : e.getFeatureSet()) {
			sum += w.get(featureIndex) * e.getFeature(featureIndex);
		}

		return sum;
	}

	/**
	 * This method sets a new lambda value given an input
	 * 
	 * @param lambda
	 *            - the input value for the new lambda
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * This method sets a new eta value given an input
	 * 
	 * @param eta
	 *            - the input value for the new eta
	 */
	public void setEta(double eta) {
		this.eta = eta;
	}

	public String toString() {
		StringBuffer buffer = new StringBuffer();

		ArrayList<Integer> temp = new ArrayList<Integer>(weights.keySet());
		Collections.sort(temp);

		for (Integer index : temp) {
			buffer.append(index + ":" + weights.get(index) + " ");
		}

		return buffer.substring(0, buffer.length() - 1);
	}
}
