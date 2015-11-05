/**
 * Created by sunlu on 11/5/15.
 * CC with IG based feature selection
 * IR factor is used for set #feature
 */


import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Random;


public class CCLDF_IG extends CC {

    private MLFeaSelect mlFeaSelect;

    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);

        // Get the IR factor for Wrapper
        double[] IRfactor = StatUtilsPro.CalcIRFactor(D);
//        System.out.println(A.toString(IRfactor));

        int L = D.classIndex();
        m_R = new Random(m_S);

        int[] indices = getChain();
        if (indices == null) {
            indices = A.make_sequence(L);
            A.shuffle(indices, m_R);
            setChain(indices);
        }

        // IG-based feature selection
        mlFeaSelect = new MLFeaSelect(L);
        mlFeaSelect.setFilterIG(true);
        Instances[] newD = mlFeaSelect.feaSelect1IR(D, IRfactor);
        nodes = new CNode[L];
        int[] pa = new int[]{};

        for (int j : m_Chain) {
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