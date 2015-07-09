/*
 *
 * ****************
 * Copyright 2015 Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ******************
 */

package it.cnr.isti.hlt.nlp4sparkml.data;

import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.Serializable;

/**
 * This is the representation of a point or document in a multiclass/binary
 * problem.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class MultilabelPoint implements Serializable {

    /**
     * The document unique ID.
     */
    private final int pointID;

    /**
     * The total number of features used. The features IDs go from 0 to
     * "numFeatures" -1.
     */
    private final int numFeatures;

    /**
     * The set of features representing this point.
     */
    private final int[] features;

    /**
     * The set of labels assigned to this point or an empty set if no labels are assigned.
     */
    private final int[] labels;

    /**
     * The set of weights associated with the features.
     */
    private final double[] weights;

    public MultilabelPoint(int pointID, int numFeatures, int[] features, double[] weights, int[] labels) {
        Cond.requireNotNull(features, "features");
        Cond.requireNotNull(weights, "weights");
        Cond.requireNotNull(labels, "labels");
        this.pointID = pointID;
        this.numFeatures = numFeatures;
        this.features = features;
        this.weights = weights;
        this.labels = labels;
    }

    /**
     * Get the set of features of this point.
     *
     * @return The set of features of this point.
     */
    public SparseVector getFeaturesAsSparseVector() {
        SparseVector v = (SparseVector) Vectors.sparse(numFeatures, features, weights);
        return v;
    }

    /**
     * Get the total number of features used for processing data.
     *
     * @return The total number of features used for processing data.
     */
    public int getNumFeatures() {
        return numFeatures;
    }

    /**
     * Get the set of valid features for this point.
     *
     * @return The set of valid features of this point.
     */
    public int[] getFeatures() {
        return features;
    }

    /**
     * Get the feature weights.
     *
     * @return The feature weights.
     */
    public double[] getWeights() {
        return weights;
    }

    /**
     * Get the set of labels assigned to this point. In binary problems, a point can have assigned
     * at most 1 label (labelID equals to 0).
     *
     * @return The set of labels assigned to this point or an empty set if no labels are assigned to it.
     */
    public int[] getLabels() {
        return labels;
    }

    /**
     * Get the point unique ID. Every point in a {@link org.apache.spark.api.java.JavaRDD} must have
     * an unique assigned ID.
     *
     * @return The point unique ID.
     */
    public int getPointID() {
        return pointID;
    }
}
