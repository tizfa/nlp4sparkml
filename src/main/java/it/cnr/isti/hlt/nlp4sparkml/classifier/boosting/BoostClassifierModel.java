/*
 *
 * ****************
 * This file is part of nlp4sparkml software package (https://github.com/tizfa/nlp4sparkml).
 *
 * Copyright 2016 Tiziano Fagni (tiziano.fagni@isti.cnr.it)
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

/*
 *
 * ****************
 * This file is part of nlp4sparkml software package (https://github.com/tizfa/nlp4sparkml).
 *
 * Copyright 2016 Tiziano Fagni (tiziano.fagni@isti.cnr.it)
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

import it.cnr.isti.hlt.nlp4sparkml.classifier.MultilabelClassifierModel;
import it.cnr.isti.hlt.nlp4sparkml.data.MultilabelPoint;
import it.cnr.isti.hlt.nlp4sparkml.data.PointClassificationResults;
import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.Estimator;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class BoostClassifierModel extends MultilabelClassifierModel<BoostClassifierModel> {

    private final BoostClassifier bc;

    private Broadcast<BoostClassifier> bcModel;

    public BoostClassifierModel(Estimator parent, BoostClassifier bc, int numFeatures) {
        super(parent, numFeatures);
        Cond.requireNotNull(bc, "bc");
        this.bc = bc;
    }


    @Override
    protected void initBroadcastVariables(JavaSparkContext sc) {
        bcModel = sc.broadcast(bc);
    }

    @Override
    protected PointClassificationResults classifyPoint(MultilabelPoint inputPoint) {
        BoostClassifier cl = bcModel.getValue();
        return cl.classify(inputPoint);
    }

}
