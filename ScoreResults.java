package score;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

/** Calculates accuracy of bird sighting predictions for Fall 2016 CS6240 course project. */
public class ScoreResults {
	public static void main(String[] args) {
		if (args.length != 2) {
			System.err.println("Usage:\nscore.ScoreResults <result file> <truth file>");
			System.exit(1);
		}
		File resultFile = new File(args[0]);
		if (!(resultFile.exists() && resultFile.isFile())) {
			System.err.println("Result file does not exist: " + args[0]);
			System.exit(1);
		}
		File truthFile = new File(args[1]);
		if (!(truthFile.exists() && truthFile.isFile())) {
			System.err.println("Truth file does not exist: " + args[1]);
			System.exit(1);
		}

		BufferedReader resultReader = null, truthReader = null;
		try {
			resultReader = new BufferedReader(new InputStreamReader(new FileInputStream(resultFile)));
			truthReader = new BufferedReader(new InputStreamReader(new FileInputStream(truthFile)));
			// Skip headers.
			resultReader.readLine(); truthReader.readLine();
			int count = 0, correct = 0;
			String resultLine, truthLine;
			while ((resultLine = resultReader.readLine()) != null && (truthLine = truthReader.readLine()) != null) {
				// Remove whitespace & parse.
				String[] resultFields = resultLine.replaceAll("\\s","").split(",");
				String[] truthFields = truthLine.replaceAll("\\s","").split(",");
				if (resultFields.length != 2 || truthFields.length != 2) {
					throw new Exception("Wrong number of fields; result:\n" + resultLine + "\nor truth:\n" + truthLine);
				}
				if (!resultFields[0].equals(truthFields[0])) {
					throw new Exception("Event Id mismatch; result: " + resultFields[0] + " vs. truth: " + truthFields[0]);
				}
				if (resultFields[1].equals(truthFields[1])) {
					correct++;
				}
				count++;
			}

			System.out.format("Accuracy= %.6f", (float)correct / (float)count);

		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
		finally {
			try {if (resultReader != null) resultReader.close();} catch (IOException e) {}
			try {if (truthReader != null) truthReader.close();} catch (IOException e) {}
		}
	}
}
