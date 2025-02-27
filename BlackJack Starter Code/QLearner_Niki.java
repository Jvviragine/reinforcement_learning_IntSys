import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class QLearner_Niki {

    private static double alpha = 0.1;
    private static double gamma = 0.99;

    //those are if we decide to use the epsilon greedy policy
    private static double epsilon = 1.0;
    private static double epsilonDecay = 0.9999;
    private static double min_epsilon = 0.001;

    private static HashMap<String, double[]> QTable;
    private static Random random = new Random();

    public static void main(String[] args) {
        BlackJackEnv game = new BlackJackEnv(BlackJackEnv.NONE);
		//Init your QTable
		QTable = new HashMap<String, double[]>();      
		
        //Variables to measure and report average performance
		double totalReward = 0.0;
        int numberOfGames = 0;
        while (notDone()) {
        	// Make sure the playOneGame method returns the end-reward of the game
            totalReward += playOneGame(game,QTable);
            numberOfGames++;

            if (epsilon > min_epsilon) {
                epsilon *= epsilonDecay;
            }

            if ((numberOfGames % 10000) == 0)
                System.out.println("Avg reward after " + numberOfGames + " games = " + 
                						(totalReward / ++numberOfGames));
        }
        // Show the learned QTable
        outputQTable(QTable);
        //exportQTableToCSV(QTable, "qtable_player_dealer.csv");
    }

    private static double playOneGame(BlackJackEnv game, HashMap<String, double[]> QTable) {
    	
    	/*You will probably require a loop
    	You will need to compute/select/find/fetch s,a,s' and r
    	Then update the right values in the QTable
    	// Don't forget to return the outcome/reward of the game
        */
        ArrayList<String> gameState = game.reset();
        double finalReward = 0;
        
        while (gameState.get(0).equals("false")) { // while game is not over
            // Get current state info
            List<String> playerCards = BlackJackEnv.getPlayerCards(gameState);
            List<String> dealerCards = BlackJackEnv.getDealerCards(gameState);
            
            int playerValue = BlackJackEnv.totalValue(playerCards);
            int dealerValue = BlackJackEnv.valueOf(dealerCards.get(0));

            boolean hasUsableAcePlayer = BlackJackEnv.holdActiveAce(playerCards);
            boolean HasUsableAceDealer = BlackJackEnv.holdActiveAce(dealerCards);
            
            // Create state key
            String stateKey = playerValue + "," + dealerValue + "," + hasUsableAcePlayer + "," + HasUsableAceDealer;
            
            // Initialize Q-values if state not seen before
            if (!QTable.containsKey(stateKey)) {
                QTable.put(stateKey, new double[]{0.0, 0.0}); // [HIT, STAND]
            }
            
            int action;

            //epsolon greedy exploitation vs exploration
            if (random.nextDouble() < epsilon) {
                // choose random - hit or stand
                action = random.nextInt(2);
            } else {
                // choose action with max Q-value
                action = QTable.get(stateKey)[0] > QTable.get(stateKey)[1] ? 0 : 1;
            }

            // Take action and observe new state and reward
            ArrayList<String> nextState = game.step(action);
            double reward = Double.parseDouble(nextState.get(1));
            
            // Update Q-value
            if (nextState.get(0).equals("true")) { // If game is over
                // just formula without next max Q since we are at terminal state
                QTable.get(stateKey)[action] = QTable.get(stateKey)[action] + 
                    alpha * (reward - QTable.get(stateKey)[action]);
                finalReward = reward;
            } else {
                // Get next state info for Q-learning update
                List<String> nextPlayerCards = BlackJackEnv.getPlayerCards(nextState);
                List<String> nextDealerCards = BlackJackEnv.getDealerCards(nextState);
                
                int nextPlayerValue = BlackJackEnv.totalValue(nextPlayerCards);
                int nextDealerValue = BlackJackEnv.valueOf(nextDealerCards.get(0)); 
                boolean nextHasUsableAcePlayer = BlackJackEnv.holdActiveAce(nextPlayerCards);
                boolean nextHasUsableAceDealer = BlackJackEnv.holdActiveAce(nextDealerCards);
                
                String nextStateKey = nextPlayerValue + "," + nextDealerValue + "," + nextHasUsableAcePlayer + "," + nextHasUsableAceDealer;
                
                if (!QTable.containsKey(nextStateKey)) {
                    QTable.put(nextStateKey, new double[]{0.0, 0.0});
                }
                
                double maxNextQ = Math.max(QTable.get(nextStateKey)[0], QTable.get(nextStateKey)[1]);
                // Apply formula : new_val = old_val + alpha * (reward + gamma * next_max - old_val)
                QTable.get(stateKey)[action] = QTable.get(stateKey)[action] + 
                    alpha * (reward + gamma * maxNextQ - QTable.get(stateKey)[action]);
            }
            
            gameState = nextState;
        }

        return finalReward;
    }

	// Example stopping condition: fixed number of games
    private static int episodeCounter = 0;
    private static boolean notDone() {
        episodeCounter++;
        return (episodeCounter <= 1000000);
    }

    private static void outputQTable(HashMap<String, double[]> QTable) {
        System.out.println("\nLearned Q-Table:");
        System.out.println("State (PlayerValue, DealerValue, hasUsableAcePlayer, hasUsableAceDealer) | Q(HIT) | Q(STAND)");
        System.out.println("--------------------------------------------------------");
        for (String state : QTable.keySet()) {
            System.out.printf("%s | %.3f | %.3f%n", 
                state, QTable.get(state)[0], QTable.get(state)[1]);
        }
    }

    private static void exportQTableToCSV(HashMap<String, double[]> QTable, String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
            // Write CSV header
            writer.append("State,Q(HIT),Q(STAND)\n");
            for (Map.Entry<String, double[]> entry : QTable.entrySet()) {
                writer.append(entry.getKey() + "," + entry.getValue()[0] + "," + entry.getValue()[1] + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
