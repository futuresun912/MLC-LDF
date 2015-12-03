/**
 * Created by sunlu on 12/3/15.
 * PACC with two-stage feature selection
 * IG: first stage; CFS: second stage
 */

import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.M;
import weka.core.Instances;
import weka.core.Instance;


public class PACC_LDF extends CC {


    private MLFeaSelect2 mlFeaSelect;

    // Training a PACC
    public void buildClassifier(Instances D) throws Exception {

        testCapabilities(D);
        int L = D.classIndex();

        // First-stage feature selection
        double perFea = getPerFeature(D);
        mlFeaSelect = new MLFeaSelect2(L);
        mlFeaSelect.setPercentFeature(perFea);
//        mlFeaSelect.setPercentFeature(0.5);
        mlFeaSelect.feaSelect1(D, L);

        // Learning of the polytree
        Polytree polytree = new Polytree();
        int[][] pa = polytree.polyTree(D, null);
        m_Chain = polytree.getChainOrder();

        if (getDebug()) {
            System.out.println(A.toString(m_Chain));
            System.out.println(M.toString(pa));
        }

        // Building the PACC
        nodes = new CNode[L];
        for (int j : m_Chain) {
            Instances tempD = mlFeaSelect.instTransform(D, j);
            mlFeaSelect.feaSelect2CFS(tempD, j);
            tempD = mlFeaSelect.instTransform(D, j);
            nodes[j] = new CNode(j, null, pa[j]);
            nodes[j].build(tempD, m_Classifier);
        }

    }

    // Test on a single instance deterministically
    public double[] distributionForInstance(Instance x) throws Exception {

        int L = x.classIndex();
        double[] y = new double[L];
        Instance xd;

        for (int j : m_Chain) {
            xd = mlFeaSelect.instTransform(x, j);
            y[j] = nodes[j].classify(xd, y);
        }

        return y;
    }


    // estimate the number of selected features
    protected double getPerFeature(Instances D) throws Exception {

        int L = D.classIndex();
        int n = D.numAttributes() - L;
        double perTemp = n > 1000 ? 0.1 : 0.4;
        int numTest = 4;

        mlFeaSelect = new MLFeaSelect2(L);
        mlFeaSelect.setPercentFeature(perTemp);
        mlFeaSelect.feaSelect1(D, numTest);

        int maxNum = 1;
//        double sumNum = 0.0;
        for (int j = 0; j < numTest; j ++) {
            Instances tempD = mlFeaSelect.instTransform(D, j);
            mlFeaSelect.feaSelect2CFS(tempD, j);
//            sumNum += mlFeaSelect.getNumFeaCfs(j);
            int temp = mlFeaSelect.getNumFeaCfs(j);
            maxNum = temp > maxNum ? temp : maxNum;
        }

//        System.out.println("mean = "+sumNum/5);
        System.out.println("*******************************");
        return (double)(maxNum+10) / n;
//        return (sumNum/numTest + 5) / n;
    }



}
