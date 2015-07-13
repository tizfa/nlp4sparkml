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

package it.cnr.isti.hlt.nlp4sparkml.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * A tokenizer which uses all the commonly used punctuation as a separator for splitting the text.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class PuntuactionTokenizer extends BaseUnaryTokenizer {

    private static final Pattern pattern = Pattern.compile("([\\s]+)|([\\:\\.\\,\\;\"\\<\\>\\[\\]\\{\\}\\\\/'\\\\&\\#\\*\\(\\)\\=\\?\\^\\!\\|])");

    @Override
    protected List<String> extractTokens(String tokenPrefix, String text) {
        String[] tokens = pattern.split(text.toLowerCase());
        ArrayList<String> ret = new ArrayList<>();
        for (String token : tokens) {
            if (token.isEmpty())
                continue;
            if (!tokenPrefix.isEmpty())
                ret.add(tokenPrefix + "_" + token);
            else
                ret.add(token);
        }
        return ret;
    }
}
