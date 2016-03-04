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

package it.cnr.isti.hlt.nlp4sparkml.classifier.boosting;/*
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

import it.cnr.isti.hlt.nlp4sparkml.data.MultilabelPoint;
import it.cnr.isti.hlt.nlp4sparkml.data.PointClassificationResults;
import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;

import java.io.Serializable;
import java.util.HashMap;

/**
 * A boosting classifier built with {@link AdaBoostMHLearner} or
 * {@link MpBoostLearner} classes.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class BoostClassifier implements Serializable {

    /**
     * The set of weak hypothesis of the model.
     */
    private final WeakHypothesis[] whs;

    public BoostClassifier(WeakHypothesis[] whs) {
        if (whs == null)
            throw new NullPointerException("The set of generated WHs is 'null'");
        this.whs = whs;
    }


    public PointClassificationResults classify(MultilabelPoint doc) {
        Cond.requireNotNull(doc, "point");
        int[] indices = doc.getFeaturesAsSparseVector().indices();
        HashMap<Integer, Integer> dict = new HashMap<Integer, Integer>();
        for (int idx = 0; idx < indices.length; idx++) {
            dict.put(indices[idx], indices[idx]);
        }
        int[] labels = new int[whs[0].getNumLabels()];
        double[] positiveThreshold = new double[labels.length];
        for (int labelID = 0; labelID < labels.length; labelID++) {
            labels[labelID] = labelID;
            positiveThreshold[labelID] = 0;
        }

        double[] scores = new double[whs[0].getNumLabels()];
        for (int i = 0; i < whs.length; i++) {
            WeakHypothesis wh = whs[i];
            for (int labelID = 0; labelID < wh.getNumLabels(); labelID++) {
                int featureID = wh.getLabelData(labelID).getFeatureID();
                if (dict.containsKey(featureID)) {
                    scores[labelID] += wh.getLabelData(labelID).getC1();
                } else {
                    scores[labelID] += wh.getLabelData(labelID).getC0();
                }
            }
        }

        PointClassificationResults res = new PointClassificationResults(doc.getPointID(), labels, scores, positiveThreshold);
        return res;
    }

}
