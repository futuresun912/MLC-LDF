
import com.sun.deploy.util.ArrayUtil;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.M;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;

import java.util.Arrays;

/**
 *  Future tasks:
 *  1. n-fold CV for depMatrix (numFold)
 *  2. How many features should be selected in the first-stage (#Feature varies among labels)
 *  3. How to compute the threshold for causalBasin (miThreshold)
 *  4. How to calculate the threshold for enhancing polytree
 *  5. Make the code simple and clear (remove the member variables of PACCsimple)
 */

/**
 * Created by sunlu on 9/11/15.
 */
public class PACCFS extends CC {


    private MLFeaSelect mlFeaSelect;

    // Training a PACC
    public void buildClassifier(Instances D) throws Exception {

        testCapabilities(D);
        int L = D.classIndex();

        // Get the IR factor for Wrapper
        double[] IRfactor = StatUtilsPro.CalcIRFactor(D);

        // First-stage feature selection
        mlFeaSelect = new MLFeaSelect(L);
        Instances[] newD = mlFeaSelect.feaSelect1(D);

        // Learning of the polytree
        Polytree polytree = new Polytree();
        polytree.setNumFolds(5);
        polytree.setDepMode(false);
        int[][] pa = polytree.polyTree(D, newD);
        m_Chain = polytree.getChainOrder();

        if (getDebug()) {
            System.out.println(A.toString(m_Chain));
            System.out.println(M.toString(pa));
        }

        // Building the PACC
//        int[] paTemp = new int[]{};
//        int[] paAll;
        nodes = new CNode[L];
        for (int j : m_Chain) {

//            paAll = Arrays.copyOf(pa[j],pa[j].length);
//
//            for (int k : paTemp){
//                int  count = 0;
//                for (int l : paAll) {
//                    if ( k != l )
//                        count ++;
//                }
//                if (count == paAll.length)
//                    paAll = A.append(paAll, k);
//            }

            // paAll saves all preceding labels,
            // while pa[j] saves only the parents in the polytree
            newD[j] = mlFeaSelect.feaSelect2(newD[j], j, pa[j], IRfactor[j]);
//            pa[j] = mlFeaSelect.getPACCpa();

            nodes[j] = new CNode(j, null, pa[j]);
            nodes[j].build(newD[j], m_Classifier);

//            paTemp = A.append(paTemp,j);
        }

    }

    // Test on a single instance deterministically
    public double[] distributionForInstance(Instance x) throws Exception {

        int L = x.classIndex();
        double[] y = new double[L];

        // Transform the test instance based on selected features
        Instance[] xd = mlFeaSelect.instTransform(x);

        for (int j : m_Chain) {
            y[j] = nodes[j].classify((Instance)xd[j].copy(), y);  // 1 or 0
        }
        return y;
    }


}