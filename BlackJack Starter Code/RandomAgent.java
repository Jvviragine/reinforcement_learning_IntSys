import java.util.ArrayList;
import java.util.List;

public class RandomAgent {

    private static BlackJackEnv game;
    private static ArrayList<String> gamestate;

    public static void main(String[] args) {
        game = new BlackJackEnv(BlackJackEnv.RENDER);
        // Playing 5 random games
        for (int i=0; i<5; i++) {
            gamestate = game.reset();
            System.out.println("The initial gamestate is: " + gamestate);
            while (gamestate.get(0).equals("false")) { // Game is not over yet
                System.out.println("The dealer is holding an " + BlackJackEnv.getDealerCards(gamestate));
                System.out.println("I am holding " + BlackJackEnv.getPlayerCards(gamestate));
                if (Math.random() > 0.2) {
                    System.out.println("I will ask an extra card");
                    gamestate = game.step(0);
                } else {
                    System.out.println("I will stand");
                    gamestate = game.step(1);
                }
                System.out.println("The gamestate passed back to me was: " + gamestate);
                System.out.println("I received a reward of " + gamestate.get(1));
            }
            System.out.println("The game ended with the dealer holding " + BlackJackEnv.getDealerCards(gamestate) +
                    " for a value of " + BlackJackEnv.totalValue(BlackJackEnv.getDealerCards(gamestate)));
            System.out.println("and me holding " + BlackJackEnv.getPlayerCards(gamestate) +
                    " for a value of " + BlackJackEnv.totalValue(BlackJackEnv.getPlayerCards(gamestate)));
            System.out.println();
        }
    }
}
