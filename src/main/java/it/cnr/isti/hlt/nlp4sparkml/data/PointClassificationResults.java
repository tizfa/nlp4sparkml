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

package it.cnr.isti.hlt.nlp4sparkml.data;

import it.cnr.isti.hlt.nlp4sparkml.utils.Cond;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class PointClassificationResults {
    private final int pointID;
    private final int[] labels;
    private final double[] scores;
    private final double[] positiveThresholds;

    public PointClassificationResults(int pointID, int[] labels, double[] scores, double[] positiveThresholds) {
        Cond.requireNotNull(labels, "labels");
        Cond.requireNotNull(scores, "scores");
        Cond.requireNotNull(positiveThresholds, "positiveThresholds");
        this.pointID = pointID;
        this.labels = labels;
        this.scores = scores;
        this.positiveThresholds = positiveThresholds;
    }

    public int getPointID() {
        return pointID;
    }

    public int[] getLabels() {
        return labels;
    }

    public double[] getScores() {
        return scores;
    }

    public double[] getPositiveThresholds() {
        return positiveThresholds;
    }
}
