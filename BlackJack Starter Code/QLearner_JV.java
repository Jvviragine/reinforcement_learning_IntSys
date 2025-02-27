import java.util.*;

public class QLearner_JV {
    private static final int NUM_STATES = 10; // Hand values from 12 to 21 (excluding terminal state)
    private static final int NUM_ACTIONS = 2; // HIT = 0, STAND = 1
    private static final double ALPHA = 0.05; // Learning rate
    private static final double GAMMA = 0.8; // Discount factor
    private static double[][] QTable = new double[NUM_STATES][NUM_ACTIONS];
    private static Random rand = new Random();
    
    public static void main(String[] args) {
        BlackJackEnv game = new BlackJackEnv(BlackJackEnv.NONE);
        double totalReward = 0.0;
        int numberOfGames = 0;

        while (notDone()) {
            totalReward += playOneGame(game);
            numberOfGames++;
            if (numberOfGames % 10000 == 0)
                System.out.println("Avg reward after " + numberOfGames + " games = " + (totalReward / numberOfGames));
        }
        
        outputQTable();
    }

    private static double playOneGame(BlackJackEnv game) {
        ArrayList<String> state = game.reset(); // Get initial game state
        int playerValue = BlackJackEnv.totalValue(BlackJackEnv.getPlayerCards(state)); // Get player hand value
        
        while (state.get(0).equals("false")) { // While the game is not over
            int stateIndex = playerValue - 12; // Normalize state (12-21 to 0-9)
            if (stateIndex < 0 || stateIndex >= NUM_STATES) break; // Invalid state

            // Choose action based on Q-values
            int action = (QTable[stateIndex][0] > QTable[stateIndex][1]) ? 0 : 1; // No randomness if values are equal

            // Execute action in the environment
            ArrayList<String> nextState = game.step(action);
            int nextPlayerValue = BlackJackEnv.totalValue(BlackJackEnv.getPlayerCards(nextState));
            int nextStateIndex = nextPlayerValue - 12;
            double reward = Double.parseDouble(nextState.get(1));

            // If terminal state, do not update future Q-values
            double maxNextQ = (nextState.get(0).equals("false") && nextStateIndex >= 0 && nextStateIndex < NUM_STATES) 
                              ? Math.max(QTable[nextStateIndex][0], QTable[nextStateIndex][1]) 
                              : 0;

            // Q-learning update
            QTable[stateIndex][action] += ALPHA * (reward + GAMMA * maxNextQ - QTable[stateIndex][action]);

            // Clip Q-values to prevent runaway updates
            QTable[stateIndex][action] = Math.max(-1.0, Math.min(1.0, QTable[stateIndex][action]));

            // Move to the next state
            state = nextState;
            playerValue = nextPlayerValue;
        }

        return Double.parseDouble(state.get(1)); // Return the final game reward
    }

    private static int episodeCounter = 0;
    private static boolean notDone() {
        episodeCounter++;
        return episodeCounter <= 1000000;
    }

    private static void outputQTable() {
        System.out.println("\nFinal Q-Table:");
        System.out.println("State (Hand Value) | HIT (0) | STAND (1)");
        for (int i = 0; i < NUM_STATES; i++) {
            System.out.printf("%2d | %.3f | %.3f\n", i + 12, QTable[i][0], QTable[i][1]);
        }
    }
}
