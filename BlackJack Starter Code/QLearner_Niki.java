import java.util.ArrayList;

public class QLearner {

    public static void main(String[] args) {
        BlackJackEnv game = new BlackJackEnv(BlackJackEnv.NONE);
		//Init your QTable
		QTable = ...        
		//Variables to measure and report average performance
		double totalReward = 0.0;
        int numberOfGames = 0;
        while (notDone()) {
        	// Make sure the playOneGame method returns the end-reward of the game
            totalReward += playOneGame(game,QTable);
            numberOfGames++;
            if ((numberOfGames % 10000) == 0)
                System.out.println("Avg reward after " + numberOfGames + " games = " + 
                						(totalReward / ++numberOfGames));
        }
        // Show the learned QTable
        outputQTable(QTable);
    }

    private static double playOneGame(BlackJackEnv game,... QTable) {
    	...
    	Your Code Here
    	...
    	You will probably require a loop
    	You will need to compute/select/find/fetch s,a,s' and r
    	Then update the right values in the QTable
    	...
    	// Don't forget to return the outcome/reward of the game
        return Double.parseDouble(finalGameState.get(1));
    }

	// Example stopping condition: fixed number of games
    private static int episodeCounter = 0;
    private static boolean notDone() {
        episodeCounter++;
        return (episodeCounter <= 1000000);
    }

    private static void outputQTable(... QTable) {
        ...
    }
}
