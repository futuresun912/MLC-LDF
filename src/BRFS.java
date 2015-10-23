
import meka.classifiers.multilabel.BR;
import meka.core.MLUtils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

/**
 * Created by sunlu on 9/15/15.
 * There is something wrong in this class.
 */
public class BRFS extends BR {

    protected Classifier m_MultiClassifiers[] = null;
    protected Instances m_InstancesTemplates[] = null;
    protected MLFeaSelect mlFeaSelect;


    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);
        int L = D.classIndex();

        // Get the Imbalance ratio-related statistics
        double[] statIR = StatUtilsPro.CalcIR(D);
        double[] IR = Arrays.copyOfRange(statIR, 0, L);
        double meanIR = statIR[L];
        double CVIR = statIR[L+1];
        if (getDebug()) {
            System.out.println("IR = "+ Arrays.toString(IR));
            System.out.println("meanIR = " + meanIR);
            System.out.println("varIR = " + CVIR);
        }


        m_MultiClassifiers = AbstractClassifier.makeCopies(m_Classifier, L);
        m_InstancesTemplates = new Instances[L];

        // First-stage feature selection
        mlFeaSelect = new MLFeaSelect(L);
        mlFeaSelect.setNumThreads(8);
        Instances[] newD = mlFeaSelect.feaSelect1(D);

        for(int j = 0; j < L; j++) {
            int[] pa = new int[]{};

            double et = Math.exp((IR[j]/Math.pow(meanIR, 0.5))*CVIR);
//            System.out.println(et);

            // f(t) = 1- e^(-t);
//            double IRfactor = 0.5*(1 - (1 / et);

            // f(t) = (e^t - 1) / (e^t + 1);
            double IRfactor = 2 * (et - 1) / (et + 1);


//            IRfactor = 2;
            System.out.println(IRfactor);

            // Second-stage feature selection
            newD[j] = mlFeaSelect.feaSelect2(newD[j], j, pa, IRfactor);

            // Remove labels except j-th
            Instances D_j = MLUtils.keepAttributesAt(new Instances(newD[j]), new int[]{j}, L);
            D_j.setClassIndex(0);

            //Build the classifier for that class
            m_MultiClassifiers[j].buildClassifier(D_j);
            m_InstancesTemplates[j] = new Instances(D_j, 0);
        }
    }

    @Override
    public double[] distributionForInstance(Instance x) throws Exception {

        int L = x.classIndex();
        double y[] = new double[L];
        Instance[] newX = mlFeaSelect.instTransform(x);

        for (int j = 0; j < L; j++) {
            Instance x_j = (Instance)newX[j].copy();
            x_j.setDataset(null);
            x_j = MLUtils.keepAttributesAt(x_j, new int[]{j}, L);
            x_j.setDataset(m_InstancesTemplates[j]);

            y[j] = m_MultiClassifiers[j].distributionForInstance(x_j)[1];
        }

        return y;
    }

}
