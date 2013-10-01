<?php
require_once 'vendor/autoload.php';
require_once 'lib/system.php';
?>

<!DOCTYPE html>
<!--[if lt IE 7]>      <html class="ie6"> <![endif]-->
<!--[if IE 7]>         <html class="ie7"> <![endif]-->
<!--[if IE 8]>         <html class="ie8"> <![endif]-->
<!--[if gt IE 8]><!--> <html>         <!--<![endif]-->

<head>
	<meta http-equiv="Content-type" content="text/html; charset=utf-8">
	<title>Page Title</title>
	<link rel="stylesheet" href="styles/styles.css" type="text/css" media="screen" title="no title" charset="utf-8">

	<script type="text/javascript" src="http://code.jquery.com/jquery-latest.js"></script>
	<script type="text/javascript" src="src/js/global.js" charset="utf-8"></script>
</head>
<body>
	<div class="wrapper">
		<form name="form" id="form" action="handler.php" method="post">
			<!-- <p><input type="text" name="site" placeholder="site"></p> -->
			<p><input type="text" name="address" placeholder="Address" value="6231 Germantown Dr"></p>
			<p><input type="text" name="citystatezip" placeholder="City, State Zip" value="Flowery Branch, GA"></p>
			<p><input type="text" name="method" placeholder="METHOD: GET, POST, PUT, DELETE"></p>
			<p><a href="#boom">Search Listing</a></p>
		</form>
		<table id="results">
			<tr id="zestimate">
				<td>Zestimate</td><td class="value"></td>
			</tr>
			<tr id="lowRange">
				<td>Low Range</td><td class="value"></td>
			</tr>
			<tr id="highRange">
				<td>High Range</td><td class="value"></td>
			</tr>
			<tr id="homeMap">
				<td>Home Map</td><td class="value"></td>
			</tr>
			<tr id="comps">
				<td>Area Comps</td><td class="value"></td>
			</tr>
		</table>
	</div>
</body>
</html>

