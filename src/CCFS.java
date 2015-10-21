
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

/**
 * Created by sunlu on 9/15/15.
 * Classifier chain with label specific features
 */
public class CCFS extends CC {

    private MLFeaSelect mlFeaSelect;

    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);
        int L = D.classIndex();
        m_R = new Random(m_S);
        mlFeaSelect = new MLFeaSelect(L);
        mlFeaSelect.setNumThreads(8);

        int[] indices = getChain();
        if (indices == null) {
            indices = A.make_sequence(L);
            A.shuffle(indices, m_R);
            setChain(indices);
        }
        System.out.println(A.toString(m_Chain));

        Instances[] newD = new Instances[L];
        // First-stage feature selection
        mlFeaSelect.setNumThreads(8);
//        mlFeaSelect.setPercentFeature(0.3);
        newD = mlFeaSelect.feaSelect1(D);
        nodes = new CNode[L];
        int[] pa = new int[]{};


        for (int j : m_Chain) {
//            newD[j] = mlFeaSelect.feaSelect2(D, j, pa);
            // Second-stage feature selection
            newD[j] = mlFeaSelect.feaSelect2(newD[j], j, pa);
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