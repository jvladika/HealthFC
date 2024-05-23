import scrapy 

from scrapy.linkextractors import LinkExtractor 
from scrapy.spiders import CrawlSpider, Rule
from scrapy.selector import Selector
from scrapy.http.request import Request

from medizin.items import AdviceItem

class MedizinSpider(scrapy.Spider):
    name = "medizinscraper"

    def start_requests(self):
        
        urls = list()
        with open("medizin/urls.txt", "r") as f:
            for line in f:
                urls.append(line.strip())
        """ 
        urls = ['https://www.medizin-transparent.at/prostatakrebs-durch-fischoel-kapseln/',
                'https://www.medizin-transparent.at/falsche-partner-wahl-durch-die-pille/',
                'https://www.medizin-transparent.at/mit-muskelkraft-gegen-rueckenschmerzen/',
                'https://www.medizin-transparent.at/zuckerkrank-durch-plastik-weichmacher/',
                'https://www.medizin-transparent.at/chemotherapie-ab-2-schwangerschafts-drittel-ungefahrlich/',
                'https://www.medizin-transparent.at/jungfraeuliche-empfaengnis-vielleicht-haeufiger-als-gedacht/',
                'https://www.medizin-transparent.at/coenzym-q10-das-mochtegern-wundermittel/',
                'https://www.medizin-transparent.at/verhuetungskettchen/',
                'https://www.medizin-transparent.at/sonnenbad-gesund-statt-krebsfordernd/',
                'https://www.medizin-transparent.at/kopfschmerzen-helfen-wirkstoffkombinationen-besser/',
                'https://www.medizin-transparent.at/hiv-test-fur-zuhause-abklarung-durch-arzt-notig/',
                'https://www.medizin-transparent.at/prostatakrebs-psa-tests-bringen-wenig/',
                'https://www.medizin-transparent.at/prostatakrebs-stoppt-lymphknoten-entfernung-ausbreitung/',
                'https://www.medizin-transparent.at/brustkrebs-exemestan-besser-als-andere-hormonblocker/',
                'https://www.medizin-transparent.at/schwungvoll-in-die-schmerzfreiheit/',
                'https://www.medizin-transparent.at/zahnspangen-muss-schoenheit-leiden/',
                'https://www.medizin-transparent.at/fluoreszierende-hirntumore-unter-dem-messer/',
                'https://www.medizin-transparent.at/religion-und-psychische-gesundheit/',
                'https://www.medizin-transparent.at/kombi-impfungen-fur-kleinkinder-angste-und-fakten/',
                'https://www.medizin-transparent.at/unfruchtbar-durch-drahtlos-internet-am-laptop/',
                'https://www.medizin-transparent.at/hilfe-gegen-graue-haare-ist-leere-behauptung/',
                'https://www.medizin-transparent.at/medikamente-oder-sport-was-hilft-besser/',
                'https://www.medizin-transparent.at/bandscheibenvorfall-abwarten-oder-operieren/',
                'https://www.medizin-transparent.at/gleichstrom-gegen-depression/',
                'https://www.medizin-transparent.at/anti-kater-pille-nur-ernuchternder-marketing-gag/',
                'https://www.medizin-transparent.at/sex-statt-kopfwehtablette/',
                'https://www.medizin-transparent.at/elektro-akupunktur-statt-schmerzmittel/',
                'https://www.medizin-transparent.at/krebs-durch-schlafmittel/',
                'https://www.medizin-transparent.at/je-mehr-freiluft-aktivitaten-umso-weniger-kurzsichtig/',
                'https://www.medizin-transparent.at/putz-sprays-raumdeos-eine-gefahr-fur-das-herz/',
                'https://www.medizin-transparent.at/krebstherapie-mit-hyperthermie-kuenstliches-fieber/',
                'https://www.medizin-transparent.at/stress-kein-hauptgrund-fuer-demenz/',
                'https://www.medizin-transparent.at/im-mai-gezeugte-kinder-zu-fruh-geboren/',
                'https://www.medizin-transparent.at/schutzt-sport-das-gehirn-vor-alkoholschaden/',
                'https://www.medizin-transparent.at/asthmarisiko-durch-pollenbelastung-fur-ungeborenes/',
                'https://www.medizin-transparent.at/wirksamkeit-von-sterol-cellulite-creme-zweifelhaft/',
                'https://www.medizin-transparent.at/mit-akupunktur-zum-nichtraucher/',
                'https://www.medizin-transparent.at/die-bunte-palette-der-osteopathie/',
                'https://www.medizin-transparent.at/das-paradox-der-brustentfernung/',
                'https://www.medizin-transparent.at/rheuma-bei-kindern-tocilizumab-als-hoffnung/',
                'https://www.medizin-transparent.at/neue-pille-tatsachlich-besser/',
                'https://www.medizin-transparent.at/vom-einzeller-in-den-selbstmord-getrieben/',
                'https://www.medizin-transparent.at/prep-hiv/',
                'https://www.medizin-transparent.at/gewichtszunahme-schwangerschaft/',
                'https://www.medizin-transparent.at/erenumab-migraene/',
                'https://www.medizin-transparent.at/antibiotika-nebenhoehlenentzuendung/',
                'https://www.medizin-transparent.at/koloskopie-das-ende-fuer-darmkrebs/',
                'https://www.medizin-transparent.at/nikotinersatz-rauchen-aufhoeren/',
                'https://www.medizin-transparent.at/hochintensitaets-training-fit-durch-minutensport/',
                'https://www.medizin-transparent.at/honig-zur-wundheilung/',
                'https://www.medizin-transparent.at/phobien-die-angst-vergessen/',
                'https://www.medizin-transparent.at/verkrummte-fingerglieder-durch-spritze-losen/',
                'https://www.medizin-transparent.at/zuckerkrank-durch-sitzen/',
                'https://www.medizin-transparent.at/diabetes-und-ubergewicht-magenoperation-als-ausweg/',
                'https://www.medizin-transparent.at/den-schmerz-ausser-acht-lassen/',
                'https://www.medizin-transparent.at/adhs-und-neurodermitis/',
                'https://www.medizin-transparent.at/mesotherapie-sticheleien-gegen-den-schmerz/',
                'https://www.medizin-transparent.at/luftverschmutzung-als-ursache-fur-autismus/',
                'https://www.medizin-transparent.at/demenz-alzheimer-vorbeugen/',
                'https://www.medizin-transparent.at/wechsel-entwarnung-fur-hormonersatztherapie/',
                'https://www.medizin-transparent.at/selen-krebs/',
                'https://www.medizin-transparent.at/sind-aepfel-im-kern-boese/',
                'https://www.medizin-transparent.at/gesundenuntersuchung/',
                'https://www.medizin-transparent.at/vitamintabletten-gesund-oder-gefahrlich/',
                'https://www.medizin-transparent.at/mythos-krebsschutz-durch-pflanzenfarbstoff-beta-carotin/',
                'https://www.medizin-transparent.at/magnete-gegen-schmerz/',
                'https://www.medizin-transparent.at/makuladegeneration-vitamine-koennten-sehverlust-bremsen/',
                'https://www.medizin-transparent.at/prostata-pflanzenpraeparate/',
                'https://www.medizin-transparent.at/cholesterinsenken-sinnlos-von-wegen/',
                'https://www.medizin-transparent.at/allergieausloser-vitamin-d/',
                'https://www.medizin-transparent.at/in-trance-gegen-darmbeschwerden-kampfen/',
                'https://www.medizin-transparent.at/nocebo-wenn-nichts-schadet/',
                'https://www.medizin-transparent.at/sex-als-ausdauersport/',
                'https://www.medizin-transparent.at/heisshunger-stop-durch-elektrische-hirnstimulation/',
                'https://www.medizin-transparent.at/schneidebretter-holz-plastik/',
                'https://www.medizin-transparent.at/macht-bier-dick/',
                'https://www.medizin-transparent.at/rheuma-linderung-durch-fischoel-fragwuerdig/',
                'https://www.medizin-transparent.at/weissdorn-herz/',
                'https://www.medizin-transparent.at/muskeln-aus-der-steckdose-ja-aber/',
                'https://www.medizin-transparent.at/glyphosat/',
                'https://www.medizin-transparent.at/wie-ungesund-ist-rotes-fleisch-fur-das-herz/',
                'https://www.medizin-transparent.at/hilfe-bei-burnout/',
                'https://www.medizin-transparent.at/milch-als-krebs-ausloeser/',
                'https://www.medizin-transparent.at/mit-laser-gegen-kurzsichtigkeit/',
                'https://www.medizin-transparent.at/mit-dem-fohn-gegen-spannungskopfweh/',
                'https://www.medizin-transparent.at/verandert-yoga-das-erbgut/',
                'https://www.medizin-transparent.at/fieber-gegen-krebs/',
                'https://www.medizin-transparent.at/elektromagnetische-wunder-fur-marlies-schild/',
                'https://www.medizin-transparent.at/wie-wirksam-ist-neuartiges-multiple-sklerose-medikament/']
        """ 

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def process_authors(self, scraped):
        result = list()
        for element in scraped:
            if element.strip() == "":
                continue
            elif "autor" in element.strip().lower():
                continue
            elif "review" in element.strip().lower():
                continue
            elif element.strip() == ",":
                continue
            else:
                result.append(element.strip())
        return result
        
    def process(self, scraped):
        result = ""
        for element in scraped:
            result += element.strip()
            result += " "
        return result

    def process_table(self, scraped):
        clean_scraped = list()

        i = 0
        while(i < len(scraped)):
            element = scraped[i].strip()
            if element not in ["", "Frage:", "Antwort:", "Erklärung:"]:
                result = element + " "
                while(i+1 < len(scraped) and element != ""):
                    i += 1
                    element = scraped[i].strip()
                    result += element + " "
                clean_scraped.append(result)
            elif element in ["Frage:", "Antwort:", "Erklärung:"]:
                clean_scraped.append(element)
            i += 1
   
        questions = list()
        answers = list()
        explanations = list()
        i = 0
        while(i+1 < len(clean_scraped)):
            element = clean_scraped[i]
            print(element)
            if element.strip() == "":
                i += 1
                continue
            elif "Frage:" in clean_scraped[i]:
                questions.append(clean_scraped[i+1].strip())
                i += 2
            elif "Antwort:" in clean_scraped[i]:
                answers.append(clean_scraped[i+1].strip())
                i += 2
            elif "Erklärung:" in clean_scraped[i]:
                explanations.append(clean_scraped[i+1].strip())
                i += 2
            else:
                i += 1

        return questions, answers, explanations
    
    def process_date(self, scraped):
        start_index = scraped.strip().index("zuletzt aktualisiert: ") + len("zuletzt aktualisiert: ")
        return scraped.strip()[start_index:].strip()

    def process_new_table(self, scraped):
        clean_scraped = list()

        for element in scraped:
            if element.strip() != "":
                clean_scraped.append(element)
        
        questions = list()
        answer_numbers = list()
        answers = list()
        explanations = list()
        i = 0
        while(i+1 < len(clean_scraped)):
            element = clean_scraped[i]
            if "Frage:" in clean_scraped[i]:
                if i+2 < len(clean_scraped):
                    if clean_scraped[i+2].strip().isnumeric():
                        questions.append(clean_scraped[i+1].strip())
                        answer_numbers.append(int(clean_scraped[i+2].strip()))
                        i += 3
                    elif "Antwort:" not in clean_scraped[i+1].strip():
                        questions.append(clean_scraped[i+1].strip())
                        answers.append(clean_scraped[i+2].strip())
                        i += 3
                    
            elif "Erklärung:" in clean_scraped[i]:
                explanations.append(clean_scraped[i+1].strip())
                i += 2
            i += 1
        
        answer_values = ['0', 'Ja, möglicherweise',
                    'Ja, wahrscheinlich',
                    'Ja',
                    'Ja, möglicherweise ein bisschen',
                    'Ja, wahrscheinlich ein bisschen',
                    'Ja, ein bisschen',
                    '7',
                    'Nein, möglicherweise nicht',
                    'Nein, wahrscheinlich nicht',
                    'Nein',
                    'Wissenschaftliche Belege fehlen']
        if len(answers) == 0:
            answers = [answer_values[num].lower() for num in answer_numbers]

        return questions, answers, explanations

    def process_old_table(self, scraped):
        clean_scraped = list()

        i = 0
        while(i < len(scraped)):
            element = scraped[i].strip()
            if element not in ["", "Frage:", "Antwort:", "Beweislage:"]:
                result = element + " "
                while(i+1 < len(scraped) and element != ""):
                    i += 1
                    element = scraped[i].strip()
                    result += element + " "
                clean_scraped.append(result)
            elif element in ["Frage:", "Antwort:", "Beweislage:"]:
                clean_scraped.append(element)
            i += 1
        
        questions = list()
        answers = list()
        explanations = list()
        i = 0
        while(i+1 < len(clean_scraped)):
            element = clean_scraped[i]
            print(element)
            if element.strip() == "":
                i += 1
                continue
            elif "Frage:" in clean_scraped[i]:
                questions.append(clean_scraped[i+1].strip())
                i += 2
            elif "Beweislage:" in clean_scraped[i]:
                if i+2 < len(clean_scraped) and clean_scraped[i+2].strip() != "":
                    final = clean_scraped[i+1].strip() + " | " + clean_scraped[i+2].strip()
                else: 
                    final = clean_scraped[i+1].strip() 
                answers.append(final)
                i += 3
            elif "Antwort:" in clean_scraped[i]:
                explanations.append(clean_scraped[i+1].strip())
                i += 2
            else:
                i += 1

        return questions, answers, explanations

    def parse(self, response):
        advice_item = AdviceItem()
        
        if "katechismus" in response.text:
            questions, answers, explanations  = self.process_new_table(response.xpath('//*[div[@class = "katechismus"]]//text()').getall())
            advice_item["text"] = self.process(response.xpath('//div[contains(@class, "pods-post-page__content")]//text()').getall())
            advice_item["studies"] = self.process(response.xpath('//div[contains(@class, "pods-post-page__study-detail")]//text()').getall())
        else:
            if "Beweislage:" in response.xpath('//table[@class="table_rating"]').get():
                questions, answers, explanations = self.process_old_table(response.xpath('//table[@class="table_rating"]//text()').getall())
            else:
                questions, answers, explanations = self.process_table(response.xpath('//table[@class="table_rating"]//text()').getall())
            advice_item["studies"] = self.process(response.xpath('//*[preceding::comment()[contains(., \'Studien im Detail\')]][following::comment()[contains(., \'Autoren\')]]//text()').getall())
            advice_item["text"] = self.process(response.xpath('//*[preceding::table[@class="table_rating"]][following::h3[contains(., "den wissenschaftlichen Studien")] or following::h4[contains(., "den wissenschaftlichen Studien")]]//text()').getall())
        

        advice_item["url"] = response.url
        advice_item["title"] = response.xpath('//title//text()').get()
        
        advice_item["date"] = self.process_date(response.xpath('//div[contains(@class, "lastedited")]//text()').get())
        advice_item["authors"] = self.process_authors(response.xpath('//*[preceding::comment()[contains(., \'Autoren\')]][following::comment()[contains(., \'SECTION END\')]][not (preceding::comment()[contains(., \'SECTION END\')])]//text()').getall())
        
        advice_item["sources"] = response.xpath("//div[@id='collapseFussnote']//text()").getall()
        advice_item["questions"] = questions
        advice_item["answers"] = answers
        advice_item["explanations"] = explanations

        return advice_item


    """    
    table =  ['\n',
           '\n',
           '\n',
           'Frage:',
           '\n',
           'Senkt die Einnahme von Magnesiumsalzen die Häufigkeit und '
           'Intensität von Muskelkrämpfen ',
           'mit unbekannter Ursache?',
           '\n',
           '\n',
           '\n',
           '\n',
           'Antwort:',
           '\n',
           '\n',
           'wahrscheinlich\xa0',
           'Nein',
           '\n',
           '\n',
           '\n',
           '\xa0',
           '\n',
           '\n',
           '\n',
           'Frage:',
           '\n',
           'Senkt die Einnahme von Magnesiumsalzen die Häufigkeit und '
           'Intensität von anstrengungsbedingten Muskelkrämpfen ',
           'beim Sport?',
           '\n',
           '\n',
           '\n',
           '\n',
           'Antwort:',
           '\n',
           '\n',
           'wissenschaftliche Belege fehlen',
           '\n',
           '\n',
           '\n',
           '\xa0',
           '\n',
           '\n',
           '\n',
           'Frage:',
           '\n',
           'Senkt die Einnahme von Magnesiumsalzen die Häufigkeit und '
           'Intensität von Muskelkrämpfen ',
           'in der Schwangerschaft?',
           '\n',
           '\n',
           '\n',
           '\n',
           'Antwort:',
           '\n',
           '\n',
           'möglicherweise\xa0',
           'Ja',
           '\n',
           '\n',
           '\n',
           'Erklärung:',
           '\n',
           'Zur Wirksamkeit von Magnesiumsalzen bei anstrengungsbedingten '
           'Muskelkrämpfen, etwa bei Sportlern, finden sich keine qualitativ '
           'hochwertigen Studien. Bei Krämpfen der Beinmuskeln ohne erkennbare '
           'Ursache haben Forscher in Studien keinen Effekt von Magnesium '
           'nachweisen können, der über Placebo hinausgeht. Schwangere Frauen '
           'profitieren möglicherweise von Magnesiumpräparaten, allerdings '
           'sind diese Belege nicht sehr belastbar.\n',
           '\n',
           '\n',
           '\n'],
    """
