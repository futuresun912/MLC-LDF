/**
 * Created by sunlu on 10/25/15.
 */

import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by sunlu on 11/5/15.
 * CC with CFS based feature selection
 * pa(y_j) is set as the start set of search algorithm
 */
public class CCFS_I extends CC {

    private MLFeaSelect mlFeaSelect;

    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);

        int L = D.classIndex();
        m_R = new Random(m_S);

        int[] indices = getChain();
        if (indices == null) {
            indices = A.make_sequence(L);
            A.shuffle(indices, m_R);
            setChain(indices);
        }

        // First-stage feature selection
        mlFeaSelect = new MLFeaSelect(L);
//        Instances[] newD = mlFeaSelect.feaSelect1(D);
        mlFeaSelect.setWrapperCfs(true);
        double emptyIR = 0.0;
        Instances[] newD = new Instances[L];

        nodes = new CNode[L];
        int[] pa = new int[]{};
        for (int j : m_Chain) {
            newD[j] = mlFeaSelect.feaSelect2(D, j, pa, emptyIR);
            nodes[j] = new CNode(j, null, pa);
            nodes[j].build(newD[j], m_Classifier);
            pa = A.append(pa, j);

        }

    }

    public double[] distributionForInstance(Instance x) throws Exception {
        int L = x.classIndex();
        double[] y = new double[L];

        // Transform the test instance
        Instance[] newX = mlFeaSelect.instTransform(x);

        for (int j : m_Chain) {
            y[j] = nodes[j].classify((Instance)newX[j].copy(), y);
        }

        return y;
    }

}