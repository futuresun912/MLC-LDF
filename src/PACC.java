

import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.M;
import meka.core.StatUtils;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Collections;

/**
 * Created by sunlu on 9/15/15.
 */
public class PACC extends CC {

    // Training a PACC
    public void buildClassifier(Instances D) throws Exception {

        testCapabilities(D);
        int L = D.classIndex();

        // Get the Imbalance ratio-related statistics
        double[] statIR = StatUtilsPro.CalcIR(D);
        double[] IR = Arrays.copyOfRange(statIR, 0, L);
        double meanIR = statIR[L];
        double varIR = statIR[L+1];
        if (getDebug()) {
            System.out.println("IR = "+ Arrays.toString(IR));
            System.out.println("meanIR = " + meanIR);
            System.out.println("varIR = " + varIR);
        }

        // Learning of the polytree
        Polytree polytree = new Polytree();
        polytree.setNumFolds(5);
        polytree.setDepMode(false);
        int[][] pa = polytree.polyTree(D, null);
        m_Chain = polytree.getChainOrder();

        // Building the PACC
        nodes = new CNode[L];
        for (int j : m_Chain) {
            nodes[j] = new CNode(j, null, pa[j]);
            nodes[j].build(D, m_Classifier);
        }

        if (getDebug()) {
            System.out.println(A.toString(m_Chain));
            System.out.println(M.toString(pa));
        }
    }

    // Test on a single instance deterministically
    public double[] distributionForInstance(Instance x) throws Exception {

        int L = x.classIndex();
        double[] y = new double[L];

        for (int j : m_Chain) {
            y[j] = nodes[j].classify((Instance)x.copy(), y);  // 1 or 0
        }
        return y;
    }

}