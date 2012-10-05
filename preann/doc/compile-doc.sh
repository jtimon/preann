#! /bin/bash

#java -jar ~/programas/jar/plantuml.jar "/home/jtimon/workspace/preann/doc/current.uml" 
java -jar ~/programas/jar/plantuml.jar "/home/jtimon/workspace/preann/doc/uml.org" 
#emacs --batch --visit=uml.org --funcall plantuml-render-buffer
#emacs --batch --visit=preann-doc.org -f org-export-as-pdf



