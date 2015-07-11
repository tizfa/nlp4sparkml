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

package it.cnr.isti.hlt.nlp4sparkml.classifier.boosting;

import it.cnr.isti.hlt.nlp4sparkml.classifier.MultilabelClassifierEstimator;
import it.cnr.isti.hlt.nlp4sparkml.data.MultilabelPoint;
import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.param.Param;

/**
 * A Spark ML estimator using AdaBoost.MH as learning algorithm.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class AdaBoostMHEstimator extends MultilabelClassifierEstimator<BoostClassifierModel> {

    private final Param<Integer> numIterations;

    public AdaBoostMHEstimator() {
        numIterations = new Param<Integer>(this, "numIterations", "The number of iterations in boosting process");
        setDefault(numIterations, 200);
    }


    @Override
    protected void initBroadcastVariables(JavaSparkContext sc) {
    }

    @Override
    protected void destroyBroadcastVariables() {

    }

    @Override
    protected BoostClassifierModel buildClassifier(JavaRDD<MultilabelPoint> inputPoints, int numFeatures) {
        Cond.requireNotNull(inputPoints, "inputPoints");
        AdaBoostMHLearner learner = new AdaBoostMHLearner(new JavaSparkContext(inputPoints.context()));
        learner.setNumIterations(getNumIterations());
        BoostClassifier bc = learner.buildModel(inputPoints);
        return new BoostClassifierModel(this, bc, numFeatures);
    }

    public int getNumIterations() {
        return getOrDefault(numIterations);
    }

    public AdaBoostMHEstimator setNumIterations(int numIterations) {
        Cond.require(numIterations > 0, "The number of iterations must be greater than 0");
        set(this.numIterations, numIterations);
        return this;
    }

    public Param<Integer> numIterations() {
        return numIterations;
    }
}
