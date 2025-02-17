import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import javax.swing.*;

public class BlackJackPanel extends JPanel {

    private ArrayList<ImageIcon> icons;
    private ImageIcon cardBack;
    private ArrayList<ImageIcon> dealerHand;
    private int dealerValue;
    private ArrayList<ImageIcon> playerHand;
    private int playerValue;
    private Boolean playerAce;
    private Boolean gameOver;

     public BlackJackPanel() {
         icons = new ArrayList<ImageIcon>();
         cardBack = new ImageIcon("gifs/back.gif");
         for (String card: BlackJackEnv.deck)
            icons.add(new ImageIcon("gifs/"+card+".gif"));
         this.setBackground(new Color(0,100,0));
         JFrame frame = new JFrame();
         frame.setBounds(100,100,400,300);
         frame.setContentPane(this);
         frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
         frame.setVisible(true);
    }

    public void render(ArrayList<String> state) {
        gameOver = (state.getFirst().equals("true"));
        List<String> dealerString = BlackJackEnv.getDealerCards(state);
        dealerHand = new ArrayList<ImageIcon>();
        for (String card : dealerString)
            dealerHand.add(getCardIcon(card));
        if (gameOver)
            dealerValue = BlackJackEnv.totalValue(BlackJackEnv.getDealerCards(state));
        else
            dealerValue = BlackJackEnv.valueOf(BlackJackEnv.getDealerCards(state).getFirst());
        List<String> playerString = BlackJackEnv.getPlayerCards(state);
        playerValue = BlackJackEnv.totalValue(playerString);
        playerAce = BlackJackEnv.holdActiveAce(playerString);
        playerHand = new ArrayList<ImageIcon>();
        for (String card : playerString)
            playerHand.add(getCardIcon(card));
        this.repaint();
    }

    private ImageIcon getCardIcon(String s) {
        int index = BlackJackEnv.deck.indexOf(s);
        return icons.get(index);
    }

    public void paintComponent(Graphics g){
         g.setColor(Color.white);
         g.drawString("Dealer:",10,30);
         if (gameOver) {
             int x = 15;
             for (ImageIcon i: dealerHand) {
                 i.paintIcon(this, g, x, 40);
                 x+=25;
             }
         } else {
             dealerHand.getFirst().paintIcon(this, g, 15, 40);
             cardBack.paintIcon(this, g, 40, 40);

         }
         g.drawString("Value = " + dealerValue, 250, 70);
         g.drawString("Player:", 10, 150);
         int x = 15;
         for (ImageIcon i: playerHand) {
             i.paintIcon(this, g, x, 160);
             x+=25;
         }
         g.drawString("Value = " + playerValue, 250, 190);
         g.drawString("with " + (playerAce?"a":"no") + " usable Ace.", 260, 210);
    }
}
