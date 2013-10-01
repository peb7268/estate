<?php
/**
 *  This is the backend implementation of the cross domain
 *  ajax call.
 * 	@param $_REQUEST['site']  - the site you want to visit
 *  @param $_REQUEST['param'] - the method you want to hit it with. ( GET, POST, PUT, ect..)
 **/
define('SITE', $_REQUEST['site']);
define('METHOD', $_REQUEST['method']);

$ch 	= curl_init(SITE);
$resp 	= curl_exec($ch);
curl_close($ch);

return $resp;