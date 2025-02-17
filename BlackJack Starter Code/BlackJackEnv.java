import java.util.*;

public class BlackJackEnv {

	public static final int TEXT = 1;
	public static final int RENDER = 2;
	public static final int NONE = 0;
	public static final int HIT = 0;
	public static final int STAND = 1;
	public static final LinkedList<String> deck = new LinkedList<String>(
			Arrays.asList("aH","kH","qH","jH","0H","9H","8H","7H","6H","5H","4H","3H","2H",
			"aD","kD","qD","jD","0D","9D","8D","7D","6D","5D","4D","3D","2D",
			"aC","kC","qC","jC","0C","9C","8C","7C","6C","5C","4C","3C","2C",
			"aS","kS","qS","jS","0S","9S","8S","7S","6S","5S","4S","3S","2S"));
	private LinkedList<String> drawdeck;
	private ArrayList<String> dealer;
	private ArrayList<String> player;
	private static int vizType = NONE;
	private static int vizDelay = 300;

	public BlackJackEnv() {}
	public BlackJackEnv(int vizType) {
		BlackJackEnv.vizType = vizType;
	}
	public BlackJackEnv(int vizType, int vizDelay) {
		BlackJackEnv.vizType = vizType; BlackJackEnv.vizDelay = vizDelay;
	}
	public ArrayList<String> reset() {
		drawdeck = new LinkedList<String>();
		drawdeck.addAll(deck);
		Collections.shuffle(drawdeck);
		dealer = new ArrayList<String>();
		dealer.add(drawdeck.poll());
		player = new ArrayList<String>();
		player.add(drawdeck.poll());
		player.add(drawdeck.poll());
		while (totalValue(player) < 12)
			player.add(drawdeck.poll());
		ArrayList<String> state = new ArrayList<String>();
		state.add("false");
		state.add("0");
		state.add("Dealer"); state.addAll(dealer);
		state.add("Player"); state.addAll(player);
		if (vizType == TEXT)
			printState(state);
		else if (vizType == RENDER)
			renderState(state);
		return state;
	}

	public ArrayList<String> step(int action) {
		ArrayList<String> state = new ArrayList<String>();
		if (action == BlackJackEnv.HIT) { // HIT
			player.add(drawdeck.poll());
			if (totalValue(player) > 21) {
				state.add("true");
				state.add("-1");
			} else {
				state.add("false");
				state.add("0");
			}
			state.add("Dealer"); state.addAll(dealer);
			state.add("Player"); state.addAll(player);
		} else if (action == BlackJackEnv.STAND){ //STAND
			state.add("true");
			while (totalValue(dealer)<17)
				dealer.add(drawdeck.poll());
			if (totalValue(player) > 21)
				state.add("-1");
			else if (totalValue(dealer) > 21)
				state.add("1");
			else if (totalValue(dealer) < totalValue(player))
				state.add("1");
			else if (totalValue(dealer) == totalValue(player))
				state.add("0");
			else
				state.add("-1");
			state.add("Dealer"); state.addAll(dealer);
			state.add("Player"); state.addAll(player);
		} else // No other actions implemented (yet?)
			System.out.println("Only actions \"0\" (HIT) and \"1\" (STAND) are implemented.  Something will surely go wrong now ...");
		if (vizType == TEXT)
			printState(state);
		else if (vizType == RENDER)
			renderState(state);
		return state;
	}

	public static int totalValue(List<String> hand) {
		int sum = 0;
		int activeAces = 0;
		for (String c: hand) {
			if (valueOf(c) == 11) activeAces++;
			sum += valueOf(c);
		}
		while (sum>21 && activeAces>0) {
			sum-=10;
			activeAces--;
		}
		return sum;
	}

	public static Boolean holdActiveAce(List<String> hand) {
		int sum = 0;
		int activeAces = 0;
		for (String c: hand) {
			if (valueOf(c) == 11) activeAces++;
			sum += valueOf(c);
		}
		while (sum>21 && activeAces>0) {
			sum-=10;
			activeAces--;
		}
		return (activeAces>0);
	}

	public static void printState(ArrayList<String> state) {
		if (state.get(0).equals("true"))
			System.out.println("The game has ended");
		else
			System.out.println("The game is still going");
		System.out.println("The reward earned this step was " + state.get(1));
		System.out.println("Dealer: " + totalValue(getDealerCards(state)) +
				" Player: " + totalValue(getPlayerCards(state)));
	}

	public static void renderState(ArrayList<String> state) {
		if (!visibleState) {
			panel = new BlackJackPanel();
			visibleState = true;
		}
		panel.render(state);
		try {
			Thread.sleep(vizDelay);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}
	private static boolean visibleState = false;
	private static BlackJackPanel panel;

	public static List<String> getDealerCards(ArrayList<String> gamestate) {
		int f = gamestate.indexOf("Dealer");
		int l = gamestate.indexOf("Player");
		return gamestate.subList(f+1,l);
	}
	public static List<String> getPlayerCards(ArrayList<String> gamestate) {
		int f = gamestate.indexOf("Player");
		return gamestate.subList(f+1,gamestate.size());
	}
	public static int valueOf(String s) {
		if (s.startsWith("a")) return 11;
		else if (s.startsWith("k") || s.startsWith("q") || s.startsWith("j") || s.startsWith("0")) return 10;
		else return Character.getNumericValue(s.charAt(0));
	}

}

